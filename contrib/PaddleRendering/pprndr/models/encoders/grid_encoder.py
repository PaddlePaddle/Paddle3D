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

from enum import IntEnum

import numpy as np
import paddle
import paddle.nn as nn

from pprndr.apis import manager

try:
    import grid_encoder
except ModuleNotFoundError:
    from pprndr.cpp_extensions import grid_encoder

__all__ = ['GridEncoder']


class GridType(IntEnum):
    hash = 0
    tiled = 1


@manager.ENCODERS.add_component
class GridEncoder(nn.Layer):
    def __init__(self,
                 input_dim=3,
                 num_levels=16,
                 level_dim=2,
                 per_level_scale=2,
                 base_resolution=16,
                 log2_hashmap_size=19,
                 desired_resolution=None,
                 gridtype="hash",
                 align_corners=False):
        super(GridEncoder, self).__init__()

        # the finest resolution desired at the last level, if provided, override per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(
                np.log2(desired_resolution / base_resolution) /
                (num_levels - 1))

        self.input_dim = input_dim  # coord dims, 2 or 3
        self.num_levels = num_levels  # num levels, each level scales resolution by 2
        self.level_dim = level_dim  # num encoded channels per level
        self.per_level_scale = per_level_scale  # scales resolution by this scale at each level
        self.log2_per_level_scale = np.log2(per_level_scale)
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = GridType[gridtype]
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2**log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale**i))
            params_in_level = min(
                self.max_params,
                (resolution if align_corners else resolution + 1)**
                input_dim)  # limit max number
            params_in_level = int(
                np.ceil(params_in_level / 8) * 8)  # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = paddle.to_tensor(offsets, dtype="int32")
        self.register_buffer('offsets', offsets)

        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = self.create_parameter(
            shape=[offset, level_dim],
            dtype='float16' if level_dim % 2 == 0 else 'float32',
            default_initializer=nn.initializer.Uniform(-1e-4, 1e-4))

    def forward(self, x):
        """
        x: [..., input_dim], normalized real world positions in [0, 1]

        return: [..., num_levels * level_dim]
        """
        with paddle.amp.auto_cast(enable=False):
            x = x.reshape([-1, self.input_dim])

            outputs, _ = grid_encoder.grid_encode(
                x, self.embeddings, self.offsets, self.input_dim,
                self.level_dim, self.num_levels, self.log2_per_level_scale,
                self.base_resolution, self.gridtype, self.align_corners,
                (not self.training) or x.stop_gradient)
            outputs = outputs.transpose([1, 0, 2]).flatten(
                start_axis=1, stop_axis=2)

        return outputs
