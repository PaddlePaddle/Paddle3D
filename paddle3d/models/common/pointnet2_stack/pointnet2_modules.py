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
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""

from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.models.layers import constant_init, kaiming_normal_init

from . import pointnet2_utils


def build_local_aggregation_module(input_channels, config):
    local_aggregation_name = config.get('name', 'StackSAModuleMSG')

    if local_aggregation_name == 'StackSAModuleMSG':
        mlps = config["mlps"]
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]
        cur_layer = StackSAModuleMSG(
            radii=config["pool_radius"],
            nsamples=config["nsample"],
            mlps=mlps,
            use_xyz=True,
            pool_method='max_pool',
        )
        num_c_out = sum([x[-1] for x in mlps])
    elif local_aggregation_name == 'VectorPoolAggregationModuleMSG':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return cur_layer, num_c_out


class StackSAModuleMSG(nn.Layer):
    def __init__(self,
                 *,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 use_xyz: bool = True,
                 pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super(StackSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.LayerList()
        self.mlps = nn.LayerList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2D(
                        mlp_spec[k],
                        mlp_spec[k + 1],
                        kernel_size=1,
                        bias_attr=False),
                    nn.BatchNorm2D(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_init(m.weight)
                if m.bias is not None:
                    constant_init(m.bias, value=0)
            if isinstance(m, nn.BatchNorm2D):
                constant_init(m.weight, value=1.0)
                constant_init(m.bias, value=0)

    def forward(self,
                xyz,
                xyz_batch_cnt,
                new_xyz,
                new_xyz_batch_cnt,
                features=None,
                empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
                features)  # (M1 + M2, C, nsample)
            new_features = new_features.transpose([1, 0, 2]).unsqueeze(
                axis=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](
                new_features)  # (1, C, M1 + M2 ..., nsample)

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
            new_features = new_features.squeeze(axis=0).transpose(
                [1, 0])  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = paddle.concat(
            new_features_list, axis=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features
