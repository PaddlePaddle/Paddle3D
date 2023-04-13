#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License")
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import math
from typing import Union

import paddle
import paddle.nn as nn

from pprndr.apis import manager
from pprndr.cameras.math_functionals import Gaussians

__all__ = ["NeRFEncoder"]


@manager.ENCODERS.add_component
class NeRFEncoder(nn.Layer):
    def __init__(self,
                 min_freq: float,
                 max_freq: float,
                 num_freqs: int,
                 input_dim: int = 3,
                 include_identity: bool = True,
                 use_radian: bool = True):
        super(NeRFEncoder, self).__init__()

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.include_identity = include_identity
        self.use_radian = use_radian

    def _expected_sin(self, x_mean: paddle.Tensor, x_var: paddle.Tensor):
        """
        Estimates mean and variance of sin(z), z ~ N(x_mean, x_var).
        """

        # When the variance is wide, shrink sin towards zero.
        return paddle.exp(-0.5 * x_var) * paddle.sin(x_mean)

    @property
    def output_dim(self):
        output_dim = self.input_dim * self.num_freqs * 2
        if self.include_identity:
            output_dim += self.input_dim
        return output_dim

    def forward(self, inputs: Union[paddle.Tensor, Gaussians]):
        """
        NeRF encoding. If `inputs` is a Gaussians, the encodings will be integrated
        as proposed in mip-NeRF.
        """

        use_integrated_encoding = isinstance(inputs, Gaussians)
        if use_integrated_encoding:
            x = inputs.mean
            covariance = inputs.covariance
        else:
            x = inputs
            covariance = None

        if self.use_radian:
            x = 2. * math.pi * x
        scales = 2**paddle.linspace(self.min_freq, self.max_freq,
                                    self.num_freqs)
        scaled_x = x[..., None] * scales
        scaled_x = scaled_x.reshape((*scaled_x.shape[:-2], -1))

        if not use_integrated_encoding:
            encoded_x = paddle.sin(
                paddle.concat([scaled_x, scaled_x + .5 * math.pi], axis=-1))
        else:
            x_var = paddle.diagonal(
                covariance, axis1=-2,
                axis2=-1)[..., :, None] * scales[None, :]**2
            x_var = x_var.reshape((*x_var.shape[:-2], -1))
            encoded_x = self._expected_sin(
                paddle.concat([scaled_x, scaled_x + .5 * math.pi], axis=-1),
                paddle.concat(2 * [x_var], axis=-1))

        if self.include_identity:
            encoded_x = paddle.concat([encoded_x, x], axis=-1)

        return encoded_x
