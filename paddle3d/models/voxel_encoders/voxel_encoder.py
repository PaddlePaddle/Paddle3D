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
"""
This code is based on https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/readers/voxel_encoder.py
Ths copyright of tianweiy/CenterPoint is as follows:
MIT License [see LICENSE for details].
"""

import paddle
import paddle.nn as nn

from paddle3d.apis import manager


def get_paddings_indicator(actual_num, max_num):
    actual_num = paddle.reshape(actual_num, [-1, 1])
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1, -1]
    max_num = paddle.arange(
        0, max_num, dtype=actual_num.dtype).reshape(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


@manager.VOXEL_ENCODERS.add_component
class VoxelMean(nn.Layer):
    def __init__(self, in_channels=4):
        super(VoxelMean, self).__init__()
        self.in_channels = in_channels

    def forward(self, features, num_voxels, coors=None):
        assert self.in_channels == features.shape[-1]

        features_sum = paddle.sum(
            features[:, :, :self.in_channels], axis=1, keepdim=False)
        points_mean = features_sum / paddle.cast(
            num_voxels, features.dtype).reshape([-1, 1])

        return points_mean
