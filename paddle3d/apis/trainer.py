# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Callable, Optional, Union

import paddle
from visualdl import LogWriter

import paddle3d.env as env
from paddle3d.apis.checkpoint import Checkpoint, CheckpointABC
from paddle3d.apis.pipeline import training_step, validation_step
from paddle3d.apis.scheduler import Scheduler, SchedulerABC
from paddle3d.utils.logger import logger
from paddle3d.utils.timer import Timer


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

        batch_sampler = BatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

        if hasattr(model, 'collate_fn'):
            collate_fn = model.collate_fn
        else:
            collate_fn = getattr(dataset, 'collate_fn', None)

        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
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


class Trainer:
    """
    """

    def __init__(
            self,
            model: paddle.nn.Layer,
            optimizer: paddle.optimizer.Optimizer,
            iters: Optional[int] = None,
            epochs: Optional[int] = None,
            train_dataset: Optional[paddle.io.Dataset] = None,
            val_dataset: Optional[paddle.io.Dataset] = None,
            resume: bool = False,
            # TODO: Default parameters should not use mutable objects, there is a risk
            checkpoint: Union[dict, CheckpointABC] = dict(),
            scheduler: Union[dict, SchedulerABC] = dict(),
            dataloader_fn: Union[dict, Callable] = dict(),
            use_amp: bool = False,
            amp_level: str = 'O1'):

        self.model = model
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.amp_level = amp_level

        _dataloader_build_fn = default_dataloader_build_fn(
            **dataloader_fn) if isinstance(dataloader_fn,
                                           dict) else dataloader_fn

        self.train_dataloader = _dataloader_build_fn(train_dataset, self.model)
        self.eval_dataloader = _dataloader_build_fn(
            val_dataset, self.model) if val_dataset else None
        self.val_dataset = val_dataset

        self.resume = resume
        vdl_file_name = None
        self.iters_per_epoch = len(self.train_dataloader)

        if iters is None:
            self.epochs = epochs
            self.iters = epochs * self.iters_per_epoch
            self.train_by_epoch = True
        else:
            self.iters = iters
            self.epochs = (iters - 1) // self.iters_per_epoch + 1
            self.train_by_epoch = False

        self.cur_iter = 0
        self.cur_epoch = 0

        if self.optimizer.__class__.__name__ == 'OneCycleAdam':
            self.optimizer.before_run(max_iters=self.iters)

        self.checkpoint = default_checkpoint_build_fn(
            **checkpoint) if isinstance(checkpoint, dict) else checkpoint

        if isinstance(scheduler, dict):
            scheduler.setdefault('train_by_epoch', self.train_by_epoch)
            scheduler.setdefault('iters_per_epoch', self.iters_per_epoch)
            self.scheduler = default_scheduler_build_fn(**scheduler)
        else:
            self.scheduler = scheduler

        if self.checkpoint is None:
            return

        if not self.checkpoint.empty:
            if not resume:
                raise RuntimeError(
                    'The checkpoint {} is not emtpy! Set `resume=True` to continue training or use another dir as checkpoint'
                    .format(self.checkpoint.rootdir))

            if self.checkpoint.meta.get(
                    'train_by_epoch') != self.train_by_epoch:
                raise RuntimeError(
                    'Unable to resume training since the train_by_epoch is inconsistent with that saved in the checkpoint'
                )

            params_dict, opt_dict = self.checkpoint.get()
            self.model.set_dict(params_dict)
            self.optimizer.set_state_dict(opt_dict)
            self.cur_iter = self.checkpoint.meta.get('iters')
            self.cur_epoch = self.checkpoint.meta.get('epochs')
            self.scheduler.step(self.cur_iter)

            logger.info(
                'Resume model from checkpoint {}, current iter set to {}'.
                format(self.checkpoint.rootdir, self.cur_iter))
            vdl_file_name = self.checkpoint.meta['vdl_file_name']
        elif resume:
            logger.warning(
                "Attempt to restore parameters from an empty checkpoint")

        if env.local_rank == 0:
            self.log_writer = LogWriter(
                logdir=self.checkpoint.rootdir, file_name=vdl_file_name)
            self.checkpoint.record('vdl_file_name',
                                   os.path.basename(self.log_writer.file_name))
            self.checkpoint.record('train_by_epoch', self.train_by_epoch)

    def train(self):
        """
        """

        sync_bn = (getattr(self.model, 'sync_bn', False) and env.nranks > 1)
        if sync_bn:
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)

        if self.use_amp:
            logger.info("Training with AMP Level {}".format(self.amp_level))
            scaler = paddle.amp.GradScaler(enable=True, init_loss_scaling=1024)
            # need to decorate model and optim in level O2
            if self.amp_level == 'O2':
                self.model, self.optimizer = paddle.amp.decorate(
                    models=self.model,
                    optimizers=self.optimizer,
                    level=self.amp_level,
                    master_weight=True)
        else:
            scaler = None

        model = self.model
        if env.nranks > 1:
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()
            model = paddle.DataParallel(self.model)

        loss_sum = 0
        timer = Timer(iters=self.iters - self.cur_iter)

        while self.cur_iter < self.iters:

            for sample in self.train_dataloader:
                self.cur_iter += 1

                if self.cur_iter % self.iters_per_epoch == 1:
                    self.cur_epoch += 1

                if self.cur_iter > self.iters:
                    break

                lr = self.optimizer.get_lr()
                loss = training_step(model, self.optimizer, scaler,
                                     self.amp_level, sample, self.cur_iter)
                loss_sum += loss.numpy()[0]

                timer.step()
                status = self.scheduler.step()

                if status.do_log and env.local_rank == 0:
                    loss_sum = float(loss_sum / self.scheduler.log_interval)
                    logger.info(
                        '[TRAIN] epoch={}/{}, iter={}/{}, loss={:.6f}, lr={:.6f} | ETA {}'
                        .format(self.cur_epoch, self.epochs, self.cur_iter,
                                self.iters, loss_sum, lr, timer.eta))

                    self.log_writer.add_scalar(
                        tag='Training/learning_rate',
                        value=lr,
                        step=self.cur_iter)
                    self.log_writer.add_scalar(
                        tag='Training/loss', value=loss_sum, step=self.cur_iter)

                    loss_sum = 0

                if status.do_eval and env.local_rank == 0:
                    # TODO: whether to save a checkpoint based on the metric
                    metrics = self.evaluate()
                    for k, v in metrics.items():
                        if not isinstance(v, paddle.Tensor) or v.numel() != 1:
                            continue

                        self.log_writer.add_scalar(
                            tag='Evaluation/{}'.format(k),
                            value=float(v),
                            step=self.cur_iter)

                if status.save_checkpoint and env.local_rank == 0:
                    if self.train_by_epoch:
                        tag = 'epoch_{}'.format(self.cur_epoch)
                    else:
                        tag = 'iter_{}'.format(self.cur_iter)

                    self.checkpoint.push(
                        tag=tag,
                        params_dict=self.model.state_dict(),
                        opt_dict=self.optimizer.state_dict(),
                        verbose=True)

                    self.checkpoint.record('iters', self.cur_iter)
                    self.checkpoint.record('epochs', self.cur_epoch)

        logger.info('Training is complete.')

        if env.local_rank == 0:
            if self.train_by_epoch:
                tag = 'epoch_{}'.format(self.epochs)
            else:
                tag = 'iter_{}'.format(self.iters)

            if not self.checkpoint.have(tag):
                self.checkpoint.push(
                    tag=tag,
                    params_dict=self.model.state_dict(),
                    opt_dict=self.optimizer.state_dict(),
                    verbose=True)

            self.checkpoint.record('iters', self.iters)
            self.checkpoint.record('epochs', self.epochs)

    def evaluate(self) -> float:
        """
        """
        sync_bn = (getattr(self.model, 'sync_bn', False) and env.nranks > 1)
        if sync_bn:
            self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model)

        if self.val_dataset is None:
            raise RuntimeError('No evaluation dataset specified!')
        msg = 'evaluate on validate dataset'
        metric_obj = self.val_dataset.metric

        for idx, sample in logger.enumerate(self.eval_dataloader, msg=msg):
            result = validation_step(self.model, sample)
            metric_obj.update(predictions=result, ground_truths=sample)

        metrics = metric_obj.compute(verbose=True)
        return metrics
