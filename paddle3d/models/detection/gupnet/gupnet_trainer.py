from cmath import log
import os
import sys
from typing import Callable, Optional, Union
from tqdm import tqdm
import paddle
from visualdl import LogWriter
import numpy as np
import paddle3d.env as env
from paddle3d.apis.checkpoint import Checkpoint, CheckpointABC
from paddle3d.apis.pipeline import validation_step
from paddle3d.apis.scheduler import Scheduler
from paddle3d.models.detection.gupnet.gupnet_scheduler import build_lr_scheduler, build_optimizer
from paddle3d.utils.logger import Logger, logger
from paddle3d.utils.shm_utils import _get_shared_memory_size_in_M
from paddle3d.models.detection.gupnet.gupnet_loss import Hierarchical_Task_Learning, GupnetLoss


def default_dataloader_build_fn(**kwargs) -> paddle.io.DataLoader:
    """
    """
    def _generate_loader(dataset: paddle.io.Dataset, model: paddle.nn.Layer):
        args = kwargs.copy()
        batch_size = args.pop('batch_size', 1)
        shuffle = False if not dataset.is_train_mode else True
        drop_last = args.pop('drop_last',
                             False if not dataset.is_train_mode else True)

        if dataset.is_train_mode:
            BatchSampler = paddle.io.DistributedBatchSampler
        else:
            # Do eval in single device
            BatchSampler = paddle.io.BatchSampler

        batch_sampler = BatchSampler(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     drop_last=drop_last)

        if hasattr(model, 'collate_fn'):
            collate_fn = model.collate_fn
        else:
            collate_fn = getattr(dataset, 'collate_fn', None)

        # DataLoader do not start sub-process in Windows and Mac
        # system, do not need to use shared memory
        use_shared_memory = sys.platform not in ['win32', 'darwin']
        # check whether shared memory size is bigger than 1G(1024M)
        if use_shared_memory:
            shm_size = _get_shared_memory_size_in_M()
            if shm_size is not None and shm_size < 1024.:
                logger.warning("Shared memory size is less than 1G, "
                               "disable shared_memory in DataLoader")
                use_shared_memory = False

        return paddle.io.DataLoader(dataset=dataset,
                                    batch_sampler=batch_sampler,
                                    collate_fn=collate_fn,
                                    use_shared_memory=use_shared_memory,
                                    **args)

    return _generate_loader


def default_checkpoint_build_fn(**kwargs) -> Checkpoint:
    """
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('save_dir', 'output')
    kwargs.setdefault('keep_checkpoint_max', 5)
    kwargs.setdefault('overwrite', True)
    return Checkpoint(**kwargs)


def default_scheduler_build_fn(**kwargs) -> Scheduler:
    """
    """
    kwargs = kwargs.copy()
    kwargs.setdefault('log_interval', 10)
    kwargs.setdefault('do_eval', False)

    if kwargs.get('train_by_epoch'):
        kwargs.setdefault('save_interval', 5)
    else:
        kwargs.setdefault('save_interval', 1000)

    return Scheduler(**kwargs)


class GupTrainer:
    """
    """
    def __init__(
            self,
            model: paddle.nn.Layer,
            optimizer: Optional[dict] = None,
            iters: Optional[int] = None,
            epochs: Optional[int] = None,
            train_dataset: Optional[paddle.io.Dataset] = None,
            val_dataset: Optional[paddle.io.Dataset] = None,
            resume: bool = False,
            # TODO: Default parameters should not use mutable objects, there is a risk
            checkpoint: Union[dict, CheckpointABC] = dict(),
            scheduler: Optional[dict] = None,
            dataloader_fn: Union[dict, Callable] = dict(),
            trainer: Optional[dict] = None,
            amp_cfg: Optional[dict] = None):

        self.model = model
        self.epoch = 0
        self.trainer = trainer

        if checkpoint is not None:
            self.save_dir = checkpoint['save_dir']
            self.logger = Logger(output=self.save_dir)
        else:
            self.save_dir = None
            self.logger = Logger()

        _dataloader_build_fn = default_dataloader_build_fn(
            **dataloader_fn) if isinstance(dataloader_fn,
                                           dict) else dataloader_fn

        self.train_dataloader = _dataloader_build_fn(train_dataset, self.model)
        self.eval_dataloader = _dataloader_build_fn(
            val_dataset, self.model) if val_dataset else None
        self.val_dataset = val_dataset

        self.resume = resume
        self.iters_per_epoch = len(self.train_dataloader)

        if iters is None:
            self.epochs = epochs
            self.iters = epochs * self.iters_per_epoch
            self.train_by_epoch = True
        else:
            self.iters = iters
            self.epochs = (iters - 1) // self.iters_per_epoch + 1
            self.train_by_epoch = False

        # build lr & bnm scheduler
        self.lr_scheduler, self.warmup_lr_scheduler = build_lr_scheduler(
            scheduler, last_epoch=-1)

        # build optimizer
        self.optimizer, self.warmup_optimizer = build_optimizer(
            optimizer, self.lr_scheduler, self.warmup_lr_scheduler, model)

    def train(self):
        start_epoch = self.epoch
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss)
        for epoch in range(start_epoch, self.epochs):
            # train one epoch
            self.logger.info('------ TRAIN EPOCH %03d ------' % (epoch + 1))
            if self.warmup_optimizer is not None and epoch < 5:
                self.logger.info('Learning Rate: %f' %
                                 self.warmup_optimizer.get_lr())

            else:
                self.logger.info('Learning Rate: %f' % self.optimizer.get_lr())

            # reset numpy seed.
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss, self.epoch)
            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' % (key[:-4], loss_weights[key])
            self.logger.info(log_str)
            ei_loss = self.train_one_epoch(loss_weights)  # 每一个epoch的损失函数
            self.epoch += 1

            # update learning rate
            if self.warmup_optimizer is not None and epoch < 5:
                self.warmup_optimizer._learning_rate.step()
            else:
                self.optimizer._learning_rate.step()

            # evaluate
            if self.trainer['save_start'] < self.epoch and (
                    self.epoch % self.trainer['eval_frequency']) == 0:
                self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                self.evaluate()

            # save checkpoints
            if self.trainer['save_start'] < self.epoch and (
                    self.epoch % self.trainer['save_frequency']) == 0:
                save_path = self.save_dir + '/checkpoints'
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path,
                                         'checkpoint_epoch_%.4d' % self.epoch)
                save_name = '{}.pdparams'.format(save_path)
                model_state = self.model.state_dict()
                paddle.save(model_state, save_name)

                self.logger.info(f'save model to {save_name}')

        self.logger.info('Finish training!!!')
        return None

    def compute_e0_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm(total=len(self.train_dataloader),
                            leave=True,
                            desc='pre-training loss stat')
        with paddle.no_grad():
            for batch_idx, samples in enumerate(self.train_dataloader):
                inputs, calibs, coord_ranges, targets, info, sample = samples
                criterion = GupnetLoss(self.epoch)
                outputs = self.model(samples)
                _, loss_terms = criterion(outputs, targets)
                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in loss_terms.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += loss_terms[key]
                progress_bar.update()
            progress_bar.close()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch
        return disp_dict

    def train_one_epoch(self, loss_weights=None):
        self.model.train()
        disp_dict = {}
        stat_dict = {}
        for batch_idx, samples in enumerate(self.train_dataloader):
            inputs, calibs, coord_ranges, targets, info, sample = samples
            # train one batch
            criterion = GupnetLoss(self.epoch)
            outputs = self.model(samples)
            total_loss, loss_terms = criterion(outputs, targets)
            if loss_weights is not None:
                total_loss = paddle.zeros(paddle.to_tensor(1))
                for key in loss_weights.keys():
                    total_loss += loss_weights[key].detach() * loss_terms[key]
            total_loss.backward()

            # update optimizer
            if self.warmup_optimizer is not None and self.epoch < 5:
                self.warmup_optimizer.step()
                self.warmup_optimizer.clear_grad()
            else:
                self.optimizer.step()
                self.optimizer.clear_grad()

            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0
                stat_dict[key] += loss_terms[key]
            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += loss_terms[key]

            # display statistics in terminal
            if trained_batch % self.trainer['disp_frequency'] == 0:
                log_str = '[EPOCH: %03d/%03d] [BATCH: %04d/%04d]' % (
                    self.epoch + 1, self.epochs, trained_batch,
                    len(self.train_dataloader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / \
                        self.trainer['disp_frequency']
                    log_str += ' %s:%.4f,' % (key, disp_dict[key])
                    disp_dict[key] = 0  # reset statistics
                self.logger.info(log_str)

        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch

        return stat_dict

    def evaluate(self) -> float:
        """
        """
        sync_bn = (getattr(self.model, 'sync_bn', False) and env.nranks > 1)
        if sync_bn:
            sparse_conv = False
            for layer in self.model.sublayers():
                if 'sparse' in str(type(layer)):
                    sparse_conv = True
                    break
            if sparse_conv:
                self.model = paddle.sparse.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.model)
            else:
                self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.model)

        if self.val_dataset is None:
            raise RuntimeError('No evaluation dataset specified!')
        msg = 'evaluate on validate dataset'
        metric_obj = self.val_dataset.metric
        self.logger.info(msg)
        # for idx, sample in logger.enumerate(self.eval_dataloader, msg=msg):
        for _, sample in enumerate(tqdm(self.eval_dataloader)):
            result = validation_step(self.model, sample)
            metric_obj.update(predictions=result, ground_truths=sample)

        metrics = metric_obj.compute(verbose=True)
        return metrics
