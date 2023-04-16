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
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/pointnet2/pointnet2_stack/voxel_query_utils.py
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""

from typing import List

import paddle
import paddle.nn as nn

from paddle3d.ops import pointnet2_ops


def voxel_query(max_range: int, radius: float, nsample: int, xyz: paddle.Tensor, \
                new_xyz: paddle.Tensor, new_coords: paddle.Tensor, point_indices: paddle.Tensor):
    """
    Args:
        max_range: int, max range of voxels to be grouped
        nsample: int, maximum number of features in the balls
        new_coords: (M1 + M2, 4), [batch_id, z, y, x] cooridnates of keypoints
        new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        point_indices: (batch_size, Z, Y, X) 4-D tensor recording the point indices of voxels
    Returns:
        idx: (M1 + M2, nsample) tensor with the indices of the features that form the query balls
    """
    z_range, y_range, x_range = max_range
    idx = pointnet2_ops.voxel_query_wrapper(new_xyz, xyz, new_coords, point_indices, \
        radius, nsample, z_range, y_range, x_range)

    empty_ball_mask = (idx[:, 0] == -1)
    idx[empty_ball_mask] = 0

    return idx, empty_ball_mask


class VoxelQueryAndGrouping(nn.Layer):
    def __init__(self, max_range: int, radius: float, nsample: int):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
        """
        super().__init__()
        self.max_range, self.radius, self.nsample = max_range, radius, nsample

    def forward(self, new_coords: paddle.Tensor, xyz: paddle.Tensor,
                xyz_batch_cnt: paddle.Tensor, new_xyz: paddle.Tensor,
                new_xyz_batch_cnt: paddle.Tensor, features: paddle.Tensor,
                voxel2point_indices: paddle.Tensor):
        """
        Args:
            new_coords: (M1 + M2 ..., 3) centers voxel indices of the ball query
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group
            voxel2point_indices: (B, Z, Y, X) tensor of points indices of voxels

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(
        ), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape),
                                           str(new_xyz_batch_cnt))
        assert new_coords.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_coords: %s, new_xyz_batch_cnt: %s' % (str(new_coords.shape), str(new_xyz_batch_cnt))
        batch_size = xyz_batch_cnt.shape[0]

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx1, empty_ball_mask1 = voxel_query(self.max_range, self.radius,
                                             self.nsample, xyz, new_xyz,
                                             new_coords, voxel2point_indices)

        idx1 = idx1.reshape([batch_size, -1, self.nsample])
        count = 0
        for bs_idx in range(batch_size):
            idx1[bs_idx] -= count
            count += xyz_batch_cnt[bs_idx]
        idx1 = idx1.reshape([-1, self.nsample])
        idx1[empty_ball_mask1] = 0

        idx = idx1
        empty_ball_mask = empty_ball_mask1

        grouped_xyz = pointnet2_ops.grouping_operation_stack(
            xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)
        # grouped_features: (M1 + M2, C, nsample)
        grouped_features = pointnet2_ops.grouping_operation_stack(
            features, xyz_batch_cnt, idx, new_xyz_batch_cnt)

        return grouped_features, grouped_xyz, empty_ball_mask
