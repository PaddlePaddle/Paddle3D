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

from typing import List

import paddle
import paddle.nn as nn

from paddle3d.geometries import CoordMode
from paddle3d.models.base.base_detection import BaseDetectionModel


class BaseLidarModel(BaseDetectionModel):
    def __init__(self,
                 box_with_velocity: bool = False,
                 with_voxelizer: bool = False,
                 max_num_points_in_voxel: int = -1,
                 in_channels: int = None):
        super().__init__(box_with_velocity=box_with_velocity)
        self.with_voxelizer = with_voxelizer
        self.max_num_points_in_voxel = max_num_points_in_voxel
        self.in_channels = in_channels
        self.point_dim = -1

    @property
    def inputs(self) -> List[dict]:
        if self.with_voxelizer:
            points = {
                'name': 'data',
                'dtype': 'float32',
                'shape': [-1, self.point_dim]
            }
            res = [points]
        else:
            voxels = {
                'name': 'voxels',
                'dtype': 'float32',
                'shape': [-1, self.max_num_points_in_voxel, self.in_channels]
            }
            coords = {'name': 'coords', 'dtype': 'int32', 'shape': [-1, 3]}
            num_points_per_voxel = {
                'name': 'num_points_per_voxel',
                'dtype': 'int32',
                'shape': [-1]
            }
            res = [voxels, coords, num_points_per_voxel]
        return res

    @property
    def sensor(self) -> str:
        return "lidar"
