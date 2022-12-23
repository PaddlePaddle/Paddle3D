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
"""OptimizerWrapper."""

from collections import defaultdict

import paddle
from paddle.optimizer import Adam, AdamW

from paddle3d.apis import manager

from .lr_schedulers import OneCycle


@manager.OPTIMIZERS.add_component
class OneCycleAdam(object):
    """OptimizerWrapper."""

    def __init__(self,
                 learning_rate,
                 beta1,
                 beta2=0.999,
                 epsilon=1e-08,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 lazy_mode=False):

        self.optimizer = paddle.optimizer.Adam(
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters,
            grad_clip=grad_clip,
            name=name,
            lazy_mode=lazy_mode)
        self.weight_decay = weight_decay
        self._learning_rate = learning_rate
        self.beta1 = beta1
        self._grad_clip = self.optimizer._grad_clip
        self.optimizer._grad_clip = None

    def _set_beta1(self, beta1, pow):
        """_set_beta1"""
        # currently support Adam and AdamW only
        update_beta1 = beta1**pow
        self.optimizer._beta1 = beta1
        if 'beta1_pow_acc' in self.optimizer._accumulators:
            for k, v in self.optimizer._accumulators['beta1_pow_acc'].items():
                self.optimizer._accumulators['beta1_pow_acc'][k] = v.fill_(
                    update_beta1)

    def before_run(self, max_iters):
        """before_run"""
        if self._learning_rate is not None:
            self._learning_rate.before_run(max_iters)
        if self.beta1 is not None:
            self.beta1.before_run(max_iters)

    def before_iter(self, curr_iter):
        """before_iter"""
        lr = self._learning_rate.get_lr(curr_iter=curr_iter)
        self.optimizer.set_lr(lr)
        beta1 = self.beta1.get_momentum(curr_iter=curr_iter)
        self._set_beta1(beta1, pow=curr_iter + 1)

    def regularize(self):
        """regularize"""
        scale_value = 1 - self.optimizer.get_lr() * self.weight_decay
        if not isinstance(self.optimizer._param_groups[0], dict):
            for i, param in enumerate(self.optimizer._param_groups):
                param.set_value(param * scale_value)
        else:
            # optimize parameters in groups
            for param_group in self.optimizer._param_groups:
                params_grads = defaultdict(lambda: list())
                for param in param_group['params']:
                    param.set_value(param * scale_value)

    def clip_grad(self):
        if not isinstance(self.optimizer._param_groups[0], dict):
            params_grads = []
            for param in self.optimizer._param_groups:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))
        else:
            # optimize parameters in groups
            for idx, param_group in enumerate(self.optimizer._param_groups):
                params_grads = defaultdict(lambda: list())
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v
                     for k, v in param_group.items() if k != 'params'})
        self._grad_clip(params_grads)

    def after_iter(self):
        """after_iter"""
        self.clip_grad()
        self.regularize()
        self.optimizer.step()
        self.optimizer.clear_grad()

    def set_state_dict(self, optimizer):
        self.optimizer.set_state_dict(optimizer)

    def get_lr(self):
        return self.optimizer.get_lr()

    def state_dict(self):
        return self.optimizer.state_dict()


@manager.OPTIMIZERS.add_component
class AdamWOnecycle(AdamW):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 clip_grad_by_norm=None,
                 parameters=None,
                 **optim_args):
        if clip_grad_by_norm is not None:
            grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_grad_by_norm)
        self.learning_rate = learning_rate
        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            beta1=beta1,
            beta2=beta2,
            grad_clip=grad_clip,
            **optim_args)

    def step(self):
        if isinstance(self._learning_rate, OneCycle):
            self._beta1 = self._learning_rate.get_mom()[0]
        super().step()
