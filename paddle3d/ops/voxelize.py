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
from voxelize import hard_voxelize


class Voxelization(nn.Layer):
    def __init__(self, voxel_size, point_cloud_range, max_num_points_in_voxel,
                 max_voxels):
        super(Voxelization, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points_in_voxel = max_num_points_in_voxel
        if isinstance(max_voxels, (tuple, list)):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = [max_voxels, max_voxels]

    def forward(self, points):
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        batch_voxels, batch_coors, batch_num_points = [], [], []
        for point in points:
            new_point = paddle.to_tensor(
                point, place=paddle.CPUPlace(), stop_gradient=True)
            voxels, coors, num_points_per_voxel, voxels_num = \
                hard_voxelize(new_point, self.voxel_size,
                    self.point_cloud_range, self.max_num_points_in_voxel,
                    max_voxels)
            voxels = voxels[0:voxels_num, :, :]
            coors = coors[0:voxels_num, :, :]
            num_points_per_voxel = num_points_per_voxel[0:voxels_num, :, :]

            batch_voxels.append(voxels)
            batch_coors.append(coors)
            batch_num_points.append(num_points_per_voxel)

        voxels_batch = paddle.concat(batch_voxels, axis=0)
        num_points_batch = paddle.concat(batch_num_points, axis=0)

        coors_batch = []
        for i, coor in enumerate(batch_coors):
            bs_idx = paddle.full(
                shape=[coor.shape[0], 1], fill_value=i, dtype=coor.dtype)
            coor_pad = paddle.concat([bs_idx, coor], axis=1)
            coors_batch.append(coor_pad)
        coors_batch = paddle.concat(coors_batch, axis=0)

        return voxels_batch, coors_batch, num_points_batch
