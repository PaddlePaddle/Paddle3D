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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.ops import voxelize

__all__ = ['HardVoxelizer']


@manager.VOXELIZERS.add_component
class HardVoxelizer(nn.Layer):
    def __init__(self, voxel_size, point_cloud_range, max_num_points_in_voxel,
                 max_num_voxels):
        super(HardVoxelizer, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points_in_voxel = max_num_points_in_voxel
        if isinstance(max_num_voxels, (tuple, list)):
            self.max_num_voxels = max_num_voxels
        else:
            self.max_num_voxels = [max_num_voxels, max_num_voxels]

    def single_forward(self, point, max_num_voxels, bs_idx):
        voxels, coors, num_points_per_voxel, voxels_num = voxelize.hard_voxelize(
            point, self.voxel_size, self.point_cloud_range,
            self.max_num_points_in_voxel, max_num_voxels)
        voxels = voxels[0:voxels_num, :, :]
        coors = coors[0:voxels_num, :]
        num_points_per_voxel = num_points_per_voxel[0:voxels_num]

        # bs_idx = paddle.full(
        #     shape=voxels_num, fill_value=bs_idx, dtype=coors.dtype)
        # bs_idx = bs_idx.reshape([-1, 1])
        # coors_pad = paddle.concat([bs_idx, coors], axis=1)
        coors = coors.reshape([1, -1, 3])
        coors_dtype = coors.dtype
        coors = coors.cast('float32')
        coors_pad = F.pad(
            coors, [1, 0], value=bs_idx, mode='constant', data_format="NCL")
        coors_pad = coors_pad.reshape([-1, 4])
        coors_pad = coors_pad.cast(coors_dtype)
        return voxels, coors_pad, num_points_per_voxel

    def forward(self, points):
        if self.training:
            max_num_voxels = self.max_num_voxels[0]
        else:
            max_num_voxels = self.max_num_voxels[1]

        if not getattr(self, "in_export_mode", False):
            batch_voxels, batch_coors, batch_num_points = [], [], []
            for bs_idx, point in enumerate(points):
                voxels, coors_pad, num_points_per_voxel = self.single_forward(
                    point, max_num_voxels, bs_idx)
                batch_voxels.append(voxels)
                batch_coors.append(coors_pad)
                batch_num_points.append(num_points_per_voxel)

            voxels_batch = paddle.concat(batch_voxels, axis=0)
            num_points_batch = paddle.concat(batch_num_points, axis=0)
            coors_batch = paddle.concat(batch_coors, axis=0)
            return voxels_batch, coors_batch, num_points_batch
        else:
            voxels, coors_pad, num_points_per_voxel = self.single_forward(
                points, max_num_voxels, 0)
            return voxels, coors_pad, num_points_per_voxel
