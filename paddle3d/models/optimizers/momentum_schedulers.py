# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
This code is based on https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/momentum_updater.py#L268
Ths copyright of mmcv is as follows:
Apache-2.0 license [see LICENSE for details].
"""

from paddle3d.apis import manager

from .utils import annealing_cos


@manager.OPTIMIZERS.add_component
class OneCycleDecayWarmupMomentum(object):
    def __init__(self,
                 momentum_peak=0.95,
                 momentum_trough=0.85,
                 step_ratio_peak=0.4):
        self.momentum_peak = momentum_peak
        self.momentum_trough = momentum_trough
        self.step_ratio_peak = step_ratio_peak
        self.momentum_phases = []  # init momentum_phases

    def before_run(self, max_iters):
        # initiate momentum_phases
        # total momentum_phases are separated as up and down
        decay_iter_per_phase = int(self.step_ratio_peak * max_iters)
        self.momentum_phases.append(
            [0, decay_iter_per_phase, self.momentum_peak, self.momentum_trough])
        self.momentum_phases.append([
            decay_iter_per_phase, max_iters, self.momentum_trough,
            self.momentum_peak
        ])

    def get_momentum(self, curr_iter):
        for (start_iter, end_iter, start_momentum,
             end_momentum) in self.momentum_phases:
            if start_iter <= curr_iter < end_iter:
                factor = (curr_iter - start_iter) / (end_iter - start_iter)
                return annealing_cos(start_momentum, end_momentum, factor)
