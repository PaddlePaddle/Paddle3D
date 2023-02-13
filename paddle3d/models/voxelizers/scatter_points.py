# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# ------------------------------------------------------------------------
# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/scatter_points.py
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

from typing import List, Tuple
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle3d.ops.voxelize import dynamic_point_to_voxel


class DynamicScatter(nn.Layer):
    """Scatters points into voxels, used in the voxel encoder with dynamic
    voxelization.
    Note:
        The CPU and GPU implementation get the same output, but have numerical
        difference after summation and division (e.g., 5e-7).
    Args:
        voxel_size (list): list [x, y, z] size of three dimension.
        point_cloud_range (list): The coordinate range of points, [x_min,
            y_min, z_min, x_max, y_max, z_max].
        average_points (bool): whether to use avg pooling to scatter points
            into voxel.
    """

    def __init__(self, voxel_size: List, point_cloud_range: List,
                 average_points: bool):
        super(DynamicScatter, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(self, points: paddle.Tensor, coors: paddle.Tensor
                       ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Scatters points into voxels.
        Args:
            points (paddle.Tensor): Points to be reduced into voxels.
            coors (paddle.Tensor): Corresponding voxel coordinates (specifically
                multi-dim voxel index) of each points.
        Returns:
            tuple[paddle.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        reduce = 'mean' if self.average_points else 'max'
        results = dynamic_point_to_voxel(points, coors, reduce)
        (voxel_feats, voxel_coors, point2voxel_map,
         voxel_points_count) = results
        return voxel_feats, voxel_coors

    def forward(self, points: paddle.Tensor,
                coors: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Scatters points/features into voxels.
        Args:
            points (paddle.Tensor): Points to be reduced into voxels.
            coors (paddle.Tensor): Corresponding voxel coordinates (specifically
                multi-dim voxel index) of each points.
        Returns:
            tuple[paddle.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        if coors.shape[-1] == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = paddle.nonzero(coors[:, 0] == i)
                inds = inds.reshape([-1])
                voxel, voxel_coor = self.forward_single(points[inds],
                                                        coors[inds][:, 1:])
                re_voxel_coor = voxel_coor.reshape([1, -1, 3])
                coors_pad = F.pad(
                    re_voxel_coor, [1, 0],
                    value=i,
                    mode='constant',
                    data_format="NCL")
                coors_pad = coors_pad.reshape([-1, 4])
                voxel_coors.append(coors_pad)
                voxels.append(voxel)
            features = paddle.concat(voxels, axis=0)
            feature_coors = paddle.concat(voxel_coors, axis=0)

            return features, feature_coors
