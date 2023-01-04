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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Sampler(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/models/backbones_3d/f2v/sampler.py#L6
    """

    def __init__(self, mode="bilinear", padding_mode="zeros"):
        """
        Initializes module
        Args:
            mode [string]: Sampling mode [bilinear/nearest]
            padding_mode [string]: Padding mode for outside grid values [zeros/border/reflection]
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, input_features, grid):
        """
        Samples input using sampling grid
        Args:
            input_features [Tensor(N, C, H_in, W_in)]: Input feature maps
            grid [Tensor(N, H_out, W,_out 2)]: Sampling grids for image features
        Returns
            output_features [Tensor(N, C, H_out, W_out)]: Output feature maps
        """
        # Sample from grid
        output = F.grid_sample(
            x=input_features,
            grid=grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        return output
