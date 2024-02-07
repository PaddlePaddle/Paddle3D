#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ["NeuSEncoder"]


@manager.ENCODERS.add_component
class NeuSEncoder(nn.Layer):
    def __init__(self,
                 input_dims: int = 3,
                 num_freqs: int = 10,
                 include_input: bool = True,
                 log_sampling: bool = True):
        super(NeuSEncoder, self).__init__()

        self.num_freqs = float(num_freqs)  # i.e., multires
        self.max_freq_log2 = self.num_freqs - 1
        self.input_dims = input_dims
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.periodic_fns = [paddle.sin, paddle.cos]

        self.out_dim = None  # computed in creat_embedding_fn
        self.embedding_fns = self.creat_embedding_fn()

    def creat_embedding_fn(self) -> list:
        # Compute out_dim
        embed_fns = []
        d = self.input_dims
        out_dim = 0

        # Include input
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        # Set freq_bands
        max_freq = self.max_freq_log2
        num_freqs = self.num_freqs
        if self.log_sampling:
            freq_bands = 2.**paddle.linspace(0, max_freq, num_freqs)
        else:
            freq_bands = paddle.linspace(2.**0., 2**max_freq, num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.output_dim = out_dim
        return embed_fns

    def forward(self, x):
        output = []
        for embed_fn in self.embedding_fns:
            output.append(embed_fn(x))

        output = paddle.concat(output, axis=-1)
        return output
