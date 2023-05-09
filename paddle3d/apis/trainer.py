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

import copy
import os
import sys
from collections import defaultdict
from typing import Callable, Optional, Union

import paddle
from visualdl import LogWriter

import paddle3d.env as env
from paddle3d.apis.checkpoint import Checkpoint, CheckpointABC
from paddle3d.apis.pipeline import training_step, validation_step
from paddle3d.apis.scheduler import Scheduler, SchedulerABC
from paddle3d.utils.logger import Logger, logger
from paddle3d.utils.shm_utils import _get_shared_memory_size_in_M
from paddle3d.utils.timer import Timer
from paddle3d.utils.profiler import add_profiler_step
from paddle3d.utils.ema import ModelEMA


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

        return paddle.io.DataLoader(
            dataset=dataset,
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
            profiler_options: Optional[dict] = None,
            dataloader_fn: Union[dict, Callable] = dict(),
            amp_cfg: Optional[dict] = None,
            do_bind: Optional[bool] = False,
            temporal_start_epoch: Optional[int] = -1,
            use_ema: Optional[bool] = False,
            ema_cfg: Optional[dict] = {}):

        self.model = model
        self.optimizer = optimizer
        self.batchsize = dataloader_fn['batch_size']

        _dataloader_build_fn = default_dataloader_build_fn(
            **dataloader_fn) if isinstance(dataloader_fn,
                                           dict) else dataloader_fn

        self.train_dataloader = _dataloader_build_fn(train_dataset, self.model)
        self.eval_dataloader = _dataloader_build_fn(
            val_dataset, self.model) if val_dataset else None
        self.val_dataset = val_dataset

        self.profiler_options = profiler_options
        self.resume = resume
        vdl_file_name = None
        self.iters_per_epoch = len(self.train_dataloader)

        self.do_bind = do_bind
        self.temporal_start_epoch = temporal_start_epoch
        self.use_ema = use_ema

        if iters is None:
            self.epochs = epochs
            self.iters = epochs * self.iters_per_epoch
            self.train_by_epoch = True
        else:
            self.iters = iters
            self.epochs = (iters - 1) // self.iters_per_epoch + 1
            self.train_by_epoch = False

        def set_lr_scheduler_iters_per_epoch(lr_scheduler,
                                             iters_per_epoch,
                                             warmup_iters=0):
            if isinstance(lr_scheduler, paddle.optimizer.lr.LinearWarmup):
                return set_lr_scheduler_iters_per_epoch(
                    lr_scheduler.learning_rate, iters_per_epoch,
                    lr_scheduler.warmup_steps)
            elif hasattr(lr_scheduler, 'learning_rate') and isinstance(
                    lr_scheduler.learning_rate,
                    paddle.optimizer.lr.LRScheduler):
                return set_lr_scheduler_iters_per_epoch(
                    lr_scheduler.learning_rate, iters_per_epoch)

            if hasattr(lr_scheduler, 'iters_per_epoch'):
                print('set lr scheduler {} iters_per_epoch={}, warmup_iters={}'.format(lr_scheduler.__class__.__name__, \
                        iters_per_epoch, warmup_iters))
                lr_scheduler.iters_per_epoch = iters_per_epoch
                lr_scheduler.warmup_iters = warmup_iters

        if hasattr(optimizer, '_learning_rate'):
            set_lr_scheduler_iters_per_epoch(optimizer._learning_rate,
                                             self.iters_per_epoch)

        self.cur_iter = 0
        self.cur_epoch = 0

        if self.optimizer.__class__.__name__ == 'OneCycleAdam':
            self.optimizer.before_run(max_iters=self.iters)

        if checkpoint is not None:
            self.logger = Logger(output=checkpoint.get('save_dir'))
        else:
            self.logger = Logger()

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

            self.cur_iter = self.checkpoint.meta.get('iters')
            self.cur_epoch = self.checkpoint.meta.get('epochs')
            params_dict, opt_dict = self.checkpoint.get(
                ema=self.ema if self.use_ema else None, step=self.cur_epoch)
            self.model.set_dict(params_dict)
            self.optimizer.set_state_dict(opt_dict)
            self.scheduler.step(self.cur_iter)

            self.logger.info(
                'Resume model from checkpoint {}, current iter set to {}'.
                format(self.checkpoint.rootdir, self.cur_iter))
            vdl_file_name = self.checkpoint.meta['vdl_file_name']
        elif resume:
            self.logger.warning(
                "Attempt to restore parameters from an empty checkpoint")

        if env.local_rank == 0:
            self.log_writer = LogWriter(
                logdir=self.checkpoint.rootdir, file_name=vdl_file_name)
            self.checkpoint.record('vdl_file_name',
                                   os.path.basename(self.log_writer.file_name))
            self.checkpoint.record('train_by_epoch', self.train_by_epoch)

        self.scaler = None
        self.amp_cfg = None

        if amp_cfg is not None and amp_cfg['use_amp']:
            scaler_cfg_ = dict(init_loss_scaling=2.**15)
            scaler_cfg_.update(**amp_cfg.pop('scaler', dict()))
            self.scaler = paddle.amp.GradScaler(**scaler_cfg_)

            amp_cfg.pop('use_amp', False)
            self.amp_cfg = amp_cfg

            amp_cfg_ = copy.deepcopy(amp_cfg)
            amp_cfg_.pop('enable', False)
            self.model.amp_cfg_ = amp_cfg_
            self.logger.info(
                'Use AMP train, AMP config: {}, Scaler config: {}'.format(
                    amp_cfg_, scaler_cfg_))

        # training with ema
        if self.use_ema:
            ema_decay = ema_cfg.get('ema_decay', 0.9998)
            ema_decay_type = ema_cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = ema_cfg.get('cycle_epoch', -1)
            ema_black_list = ema_cfg.get('ema_black_list', None)
            ema_filter_no_grad = ema_cfg.get('ema_filter_no_grad', False)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list,
                ema_filter_no_grad=ema_filter_no_grad)

    def train(self):
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

        model = self.model
        group = None
        if env.nranks > 1:
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                group = paddle.distributed.init_parallel_env()
            model = paddle.DataParallel(self.model)

        losses_sum = defaultdict(float)
        timer = Timer(iters=self.iters - self.cur_iter)

        while self.cur_iter < self.iters:

            for sample in self.train_dataloader:
                if self.cur_iter == 1 and self.do_bind and int(
                        os.environ.get('FLAGS_selected_gpus', 0)) == 0:
                    test_cmd = "j=0 | j=$(( $j + 1 ))"
                    rst = os.system(test_cmd)
                    if rst != 0:
                        self.logger.warning(
                            "The system doesn't support i++ bash command, will not do cpu core bind"
                        )
                    else:
                        # Skip the first iter, let all necessary threads to be inited.
                        # Each thread will assign three cpu cores.
                        # IMPORTANT NOTE! If the hardware doesn't have enough cpu cores,
                        # can delete one or two `"(( i++ )) \n" \` in the line 287/288.
                        cmd = "bash \n"  \
                              "ps aux | grep \" -u tools/train.py\" | grep -v grep | awk '{print $2}' > taskset.log \n" \
                              "i=0 \n" \
                              "for pid in `cat taskset.log`; do \n" \
                              "i=$(( $i + 1 )) \n" \
                              "taskset -pc  $i,$(( i + 1 )),$(( i + 2 )) $pid \n" \
                              "i=$(( $i + 1 )) \n" \
                              "i=$(( $i + 1 )) \n" \
                              "done \n"
                        os.system(cmd)
                self.cur_iter += 1

                if self.cur_iter % self.iters_per_epoch == 1:
                    self.cur_epoch += 1

                # simple implementation of SequentialControlHook
                if self.temporal_start_epoch != -1 and (
                        self.cur_epoch > self.temporal_start_epoch):
                    model.with_prev = True
                else:
                    model.with_prev = False

                if self.cur_iter > self.iters:
                    break

                add_profiler_step(self.profiler_options)

                lr = self.optimizer.get_lr()

                output = training_step(
                    model,
                    self.optimizer,
                    sample,
                    self.cur_iter,
                    scaler=self.scaler,
                    amp_cfg=self.amp_cfg,
                    all_fused_tensors=getattr(self.optimizer,
                                              'all_fused_tensors', None),
                    group=group)

                for loss_name, loss_value in output.items():
                    losses_sum[loss_name] += float(loss_value)

                timer.step(self.batchsize)
                status = self.scheduler.step()

                if status.do_log and env.local_rank == 0:
                    loss_log = ''
                    for loss_name, loss_value in losses_sum.items():
                        loss_value = loss_value / self.scheduler.log_interval
                        loss_log += ', {}={:.6f}'.format(loss_name, loss_value)
                        self.log_writer.add_scalar(
                            tag='Training/' + loss_name,
                            value=loss_value,
                            step=self.cur_iter)

                    self.log_writer.add_scalar(
                        tag='Training/learning_rate',
                        value=lr,
                        step=self.cur_iter)

                    self.logger.info(
                        '[TRAIN] epoch={}/{}, iter={}/{} {}, lr={:.6f}, batch_cost: {:.6f} sec, ips: {:.6f} images/s | ETA {}'
                        .format(self.cur_epoch, self.epochs, self.cur_iter,
                                self.iters, loss_log, lr, timer.speed,
                                timer.ips, timer.eta))

                    losses_sum.clear()

                if self.use_ema:  # update ema_weight at each iter
                    self.ema.update()

                if status.do_eval and env.local_rank == 0:
                    # TODO: whether to save a checkpoint based on the metric
                    # if use ema, evaluation should be based on ema weights
                    # so replace current weights with ema weights
                    if self.use_ema:
                        # apply ema weight on model
                        curr_weight = copy.deepcopy(self.model.state_dict())
                        self.model.set_dict(self.ema.apply())
                    metrics = self.evaluate()
                    for k, v in metrics.items():
                        if not isinstance(v, paddle.Tensor) or v.numel() != 1:
                            continue

                        self.log_writer.add_scalar(
                            tag='Evaluation/{}'.format(k),
                            value=float(v),
                            step=self.cur_iter)

                    if self.use_ema:
                        # reset original weight
                        self.model.set_dict(curr_weight)

                if status.save_checkpoint and env.local_rank == 0:
                    if self.train_by_epoch:
                        tag = 'epoch_{}'.format(self.cur_epoch)
                    else:
                        tag = 'iter_{}'.format(self.cur_iter)

                    if self.use_ema:
                        self.checkpoint.push(
                            tag=tag,
                            params_dict=self.model.state_dict(),
                            opt_dict=self.optimizer.state_dict(),
                            verbose=True,
                            ema_model=self.ema.apply())

                    self.checkpoint.push(
                        tag=tag,
                        params_dict=self.model.state_dict(),
                        opt_dict=self.optimizer.state_dict(),
                        verbose=True)

                    self.checkpoint.record('iters', self.cur_iter)
                    self.checkpoint.record('epochs', self.cur_epoch)

        self.logger.info('Training is complete.')

        if env.local_rank == 0:
            if self.train_by_epoch:
                tag = 'epoch_{}'.format(self.epochs)
            else:
                tag = 'iter_{}'.format(self.iters)

            if not self.checkpoint.have(tag):
                if self.use_ema:
                    self.checkpoint.push(
                        tag=tag,
                        params_dict=self.model.state_dict(),
                        opt_dict=self.optimizer.state_dict(),
                        verbose=True,
                        ema_model=self.ema.apply())

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

        for idx, sample in self.logger.enumerate(self.eval_dataloader, msg=msg):
            result = validation_step(self.model, sample)
            metric_obj.update(predictions=result, ground_truths=sample)

        metrics = metric_obj.compute(verbose=True)
        return metrics
