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
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/pointnet2/pointnet2_stack/voxel_pool_modules.py
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""
from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.models.common.pointnet2_stack import voxel_query_utils
from paddle3d.models.layers import constant_init, kaiming_normal_init


class NeighborVoxelSAModuleMSG(nn.Layer):
    def __init__(self,
                 *,
                 query_ranges: List[List[int]],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 use_xyz: bool = True,
                 pool_method='max_pool'):
        """
        Args:
            query_ranges: list of int, list of neighbor ranges to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(query_ranges) == len(nsamples) == len(mlps)

        self.groupers = nn.LayerList()
        self.mlps_in = nn.LayerList()
        self.mlps_pos = nn.LayerList()
        self.mlps_out = nn.LayerList()
        for i in range(len(query_ranges)):
            max_range = query_ranges[i]
            nsample = nsamples[i]
            radius = radii[i]
            self.groupers.append(
                voxel_query_utils.VoxelQueryAndGrouping(max_range, radius,
                                                        nsample))
            mlp_spec = mlps[i]

            cur_mlp_in = nn.Sequential(
                nn.Conv1D(
                    mlp_spec[0], mlp_spec[1], kernel_size=1, bias_attr=False),
                nn.BatchNorm1D(mlp_spec[1]))

            cur_mlp_pos = nn.Sequential(
                nn.Conv2D(3, mlp_spec[1], kernel_size=1, bias_attr=False),
                nn.BatchNorm2D(mlp_spec[1]))

            cur_mlp_out = nn.Sequential(
                nn.Conv1D(
                    mlp_spec[1], mlp_spec[2], kernel_size=1, bias_attr=False),
                nn.BatchNorm1D(mlp_spec[2]), nn.ReLU())

            self.mlps_in.append(cur_mlp_in)
            self.mlps_pos.append(cur_mlp_pos)
            self.mlps_out.append(cur_mlp_out)

        self.relu = nn.ReLU()
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D) or isinstance(m, nn.Conv1D):
                kaiming_normal_init(m.weight)
                if m.bias is not None:
                    constant_init(m.bias, value=0)
            if isinstance(m, nn.BatchNorm2D) or isinstance(m, nn.BatchNorm1D):
                constant_init(m.weight, value=1.0)
                constant_init(m.bias, value=0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, \
                                        new_coords, features, voxel2point_indices):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the features
        :param point_indices: (B, Z, Y, X) tensor of point indices
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        # change the order to [batch_idx, z, y, x]
        index = paddle.to_tensor([0, 3, 2, 1], dtype='int32')
        new_coords = paddle.index_select(new_coords, index, axis=-1)
        new_features_list = []

        for k in range(len(self.groupers)):
            # features_in: (1, C, M1+M2)
            features_in = features.transpose([1, 0]).unsqueeze(0)
            features_in = self.mlps_in[k](features_in)
            # features_in: (1, M1+M2, C)
            features_in = features_in.transpose([0, 2, 1])
            # features_in: (M1+M2, C)
            features_in = features_in.reshape([-1, features_in.shape[-1]])
            # grouped_features: (M1+M2, C, nsample)
            # grouped_xyz: (M1+M2, 3, nsample)
            grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](
                new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
                features_in, voxel2point_indices)

            grouped_features[empty_ball_mask] = 0

            # grouped_features: (1, C, M1+M2, nsample)
            grouped_features = grouped_features.transpose([1, 0,
                                                           2]).unsqueeze(axis=0)
            # grouped_xyz: (M1+M2, 3, nsample)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            grouped_xyz[empty_ball_mask] = 0
            # grouped_xyz: (1, 3, M1+M2, nsample)
            grouped_xyz = grouped_xyz.transpose([1, 0, 2]).unsqueeze(0)
            # grouped_xyz: (1, C, M1+M2, nsample)
            position_features = self.mlps_pos[k](grouped_xyz)
            new_features = grouped_features + position_features
            new_features = self.relu(new_features)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features,
                    kernel_size=[1, new_features.shape[3]]).squeeze(
                        axis=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features,
                    kernel_size=[1, new_features.shape[3]]).squeeze(
                        axis=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError

            new_features = self.mlps_out[k](new_features)
            new_features = new_features.squeeze(axis=0).transpose(
                [1, 0])  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        # (M1 + M2 ..., C)
        new_features = paddle.concat(new_features_list, axis=1)
        return new_features
