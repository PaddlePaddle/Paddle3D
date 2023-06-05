# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is based on https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L432
Ths copyright of mmcv is as follows:
Apache-2.0 license [see LICENSE for details].

This code is based on https://github.com/TRAILab/CaDDN/blob/master/tools/train_utils/optimization/learning_schedules_fastai.py#L60
Ths copyright of CaDDN is as follows:
Apache-2.0 license [see LICENSE for details].
"""
from functools import partial
import math
import paddle
from paddle.optimizer.lr import LRScheduler

from paddle3d.apis import manager

from .utils import annealing_cos


@manager.LR_SCHEDULERS.add_component
class OneCycleWarmupDecayLr(LRScheduler):
    def __init__(self,
                 base_learning_rate,
                 lr_ratio_peak=10,
                 lr_ratio_trough=1e-4,
                 step_ratio_peak=0.4):
        self.base_learning_rate = base_learning_rate
        self.lr_ratio_peak = lr_ratio_peak
        self.lr_ratio_trough = lr_ratio_trough
        self.step_ratio_peak = step_ratio_peak
        self.lr_phases = []  # init lr_phases
        self.anneal_func = annealing_cos

    def before_run(self, max_iters):
        """before_run"""
        warmup_iter_per_phase = int(self.step_ratio_peak * max_iters)
        self.lr_phases.append([0, warmup_iter_per_phase, 1, self.lr_ratio_peak])
        self.lr_phases.append([
            warmup_iter_per_phase, max_iters, self.lr_ratio_peak,
            self.lr_ratio_trough
        ])

    def get_lr(self, curr_iter):
        """get_lr"""
        for (start_iter, end_iter, lr_start_ratio,
             lr_end_ratio) in self.lr_phases:
            if start_iter <= curr_iter < end_iter:
                factor = (curr_iter - start_iter) / (end_iter - start_iter)
                return self.anneal_func(
                    self.base_learning_rate * lr_start_ratio,
                    self.base_learning_rate * lr_end_ratio, factor)


class LRSchedulerCycle(LRScheduler):
    def __init__(self, total_step, lr_phases, mom_phases):

        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append((int(start * total_step),
                                       int(lr_phases[i + 1][0] * total_step),
                                       lambda_func))
            else:
                self.lr_phases.append((int(start * total_step), total_step,
                                       lambda_func))
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append((int(start * total_step),
                                        int(mom_phases[i + 1][0] * total_step),
                                        lambda_func))
            else:
                self.mom_phases.append((int(start * total_step), total_step,
                                        lambda_func))
        assert self.mom_phases[0][0] == 0
        super().__init__()


@manager.OPTIMIZERS.add_component
class OneCycle(LRSchedulerCycle):
    def __init__(self, total_step, lr_max, moms, div_factor, pct_start):
        self.lr_max = lr_max
        self.moms = moms
        self.last_moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        a1 = int(total_step * self.pct_start)
        a2 = total_step - a1
        low_lr = self.lr_max / self.div_factor
        lr_phases = ((0, partial(annealing_cos, low_lr, self.lr_max)),
                     (self.pct_start,
                      partial(annealing_cos, self.lr_max, low_lr / 1e4)))
        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        self.learning_rate = low_lr
        super().__init__(total_step, lr_phases, mom_phases)

    def get_lr(self):
        lr = self.last_lr
        for start, end, func in self.lr_phases:
            if self.last_epoch >= start:
                lr = func((self.last_epoch - start) / (end - start))
        return lr

    def set_mom(self):
        mom = self.last_moms[0]
        for start, end, func in self.mom_phases:
            if self.last_epoch >= start:
                mom = func((self.last_epoch - start) / (end - start))
        self.last_moms[0] = mom

    def step(self, epoch=None):
        super().step()
        self.set_mom()

    def get_mom(self):
        return self.last_moms


@manager.LR_SCHEDULERS.add_component
class CosineAnnealingDecayByEpoch(paddle.optimizer.lr.CosineAnnealingDecay):
    iters_per_epoch = 1
    warmup_iters = 0

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        else:
            cur_epoch = (
                self.last_epoch + self.warmup_iters) // self.iters_per_epoch
            return annealing_cos(self.base_lr, self.eta_min,
                                 cur_epoch / self.T_max)

    def _get_closed_form_lr(self):
        return self.get_lr()


@manager.LR_SCHEDULERS.add_component
class CosineWarmupMultiStepDecayByEpoch(LRScheduler):
    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 start_lr,
                 milestones,
                 decay_rate,
                 end_lr=None):
        self.iters_per_epoch = 1
        self.warmup_iters = 0
        self.warmup_epochs = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr if end_lr is not None else learning_rate
        self.milestones = milestones
        self.decay_rate = decay_rate
        super(CosineWarmupMultiStepDecayByEpoch, self).__init__(learning_rate)

    def get_lr(self):
        # update current epoch
        cur_epoch = (
            self.last_epoch + self.warmup_iters) // self.iters_per_epoch

        # cosine warmup
        if cur_epoch < self.warmup_epochs:
            return self.start_lr + (self.end_lr - self.start_lr) * (
                1 - math.cos(math.pi * cur_epoch / self.warmup_epochs)) / 2
        else:
            if self.last_epoch in [
                    self.milestones[0] * self.iters_per_epoch,
                    self.milestones[1] * self.iters_per_epoch
            ]:
                self.end_lr *= self.decay_rate
                return self.end_lr
            else:
                return self.end_lr
