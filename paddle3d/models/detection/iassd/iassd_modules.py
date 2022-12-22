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

# This code is based on https://github.com/yifanzhang713/IA-SSD/blob/main/pcdet/ops/pointnet2/pointnet2_batch/pointnet2_modules.py

import os
from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.ops import pointnet2_ops

__all__ = ["SAModuleMSG_WithSampling", "Vote_layer"]


class QueryAndGroup(nn.Layer):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: paddle.Tensor, new_xyz: paddle.Tensor,
                features: paddle.Tensor):
        """
        xyz: (B, N, 3)
        new_xyz: (B, npoint, 3)
        features: (B, C, N)
        """
        idx = pointnet2_ops.ball_query_batch(
            new_xyz, xyz, self.radius, self.nsample)  # (B, npoints, nsample)
        xyz_trans = xyz.transpose([0, 2, 1])  # (B, 3, N)
        grouped_xyz = pointnet2_ops.grouping_operation_batch(
            xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose([0, 2, 1]).unsqueeze(-1)

        if features is not None:
            grouped_features = pointnet2_ops.grouping_operation_batch(
                features, idx)
            if self.use_xyz:
                new_features = paddle.concat(
                    [grouped_xyz, grouped_features],
                    axis=1)  # (B, 3+C, npoint, nsmaple)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz
        return new_features


class SAModuleMSG_WithSampling(nn.Layer):
    def __init__(self,
                 *,
                 npoint: int,
                 sample_range: int,
                 sample_type: str,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 use_xyz: bool = True,
                 dilated_group=False,
                 pool_method='max_pool',
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_classes: int):
        """
        """
        super().__init__()
        assert len(radii) == len(nsamples)

        self.npoint = npoint
        self.sample_type = sample_type
        self.sample_range = sample_range
        self.dilated_group = dilated_group

        self.groupers = nn.LayerList()
        self.mlps = nn.LayerList()

        out_channels = 0
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if self.dilated_group:
                raise NotImplementedError
            else:
                self.groupers.append(
                    QueryAndGroup(radius, nsample, use_xyz=use_xyz))

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
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        if (aggregation_mlp is not None) and (len(aggregation_mlp) !=
                                              0) and (len(self.mlps) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1D(
                        out_channels,
                        aggregation_mlp[k],
                        kernel_size=1,
                        bias_attr=False),
                    nn.BatchNorm1D(aggregation_mlp[k]),
                    nn.ReLU()
                ])
            out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1D(
                        out_channels,
                        confidence_mlp[k],
                        kernel_size=1,
                        bias_attr=False),
                    nn.BatchNorm1D(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1D(
                    out_channels, num_classes, kernel_size=1, bias_attr=True))
            self.confidence_layer = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layer = None

    def forward(self,
                xyz: paddle.Tensor,
                features: paddle.Tensor = None,
                cls_features: paddle.Tensor = None,
                new_xyz=None,
                ctr_xyz=None):
        """
        xyz: (B, N, 3)
        features: (B, C, N)
        cls_features: (B, npoint, num_class) or None
        new_xyz: (B, npoint, 3) or None
        ctr_xyz: (B, npoint, 3) or None
        """
        new_features_list = []
        xyz_flipped = xyz.transpose([0, 2, 1])

        # Sample operation
        if ctr_xyz is None:

            # No downsample
            if xyz.shape[1] <= self.npoint:

                sample_idx = paddle.arange(
                    xyz.shape[1], dtype='int32') * paddle.ones(
                        xyz.shape[:2], dtype='int32')

            # ctr downsample
            elif 'ctr' in self.sample_type:
                cls_features_max = cls_features.max(axis=-1)
                score_pred = F.sigmoid(cls_features_max)
                sample_value, sample_idx = paddle.topk(
                    score_pred, self.npoint, axis=-1)
                sample_idx = sample_idx.astype('int32')  # (B, npoint)

            # D-FPS downsample
            elif 'D-FPS' in self.sample_type:
                sample_idx = pointnet2_ops.farthest_point_sample(
                    xyz, self.npoint)  # (B, npoint)

            new_xyz = pointnet2_ops.gather_operation(
                xyz_flipped, sample_idx).transpose([0, 2, 1])  # (B, npoint, 3)

        else:
            new_xyz = ctr_xyz

        # MSG group operation (ball query and group)
        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):
                new_features = self.groupers[i](
                    xyz, new_xyz, features)  # (B, C, npoint, nsample)
                new_features = self.mlps[i](
                    new_features)  # (B, mlp[-1], npoint, nsample)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.shape[-1]])
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.shape[-1]])
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                new_features_list.append(new_features)

            new_features = paddle.concat(
                new_features_list, axis=1)  # (B, mlp_cat, npoint)

            if self.aggregation_layer is not None:
                new_features = self.aggregation_layer(
                    new_features)  # (B, mlp_agg, npoint)

        else:
            new_features = pointnet2_ops.gather_operation(
                features, sample_idx)  # (B, C, npoint)

        # Confidence layer
        if self.confidence_layer is not None:
            cls_features = self.confidence_layer(new_features).transpose(
                [0, 2, 1])  # (B, npoint, num_class)
        else:
            cls_features = None

        return new_xyz, new_features, cls_features


class Vote_layer(nn.Layer):
    def __init__(self, mlp_list, pre_channel, max_translate_range):
        super().__init__()
        if len(mlp_list) > 0:
            shared_mlps = []
            for i in range(len(mlp_list)):
                shared_mlps.extend([
                    nn.Conv1D(
                        pre_channel,
                        mlp_list[i],
                        kernel_size=1,
                        bias_attr=False),
                    nn.BatchNorm1D(mlp_list[i]),
                    nn.ReLU()
                ])
                pre_channel = mlp_list[i]
            self.mlps = nn.Sequential(*shared_mlps)
        else:
            self.mlps = None

        self.ctr_reg = nn.Conv1D(pre_channel, 3, kernel_size=1)
        self.max_offset_limit = paddle.to_tensor(
            max_translate_range,
            dtype='float32') if max_translate_range is not None else None

    def forward(self, xyz, features):
        if self.mlps is not None:
            new_features = self.mlps(features)  # [2, 256, 256] -> [2, 128, 256]
        else:
            new_features = features

        ctr_offsets = self.ctr_reg(new_features)  # [2, 128, 256] -> [2, 3, 256]
        ctr_offsets = ctr_offsets.transpose([0, 2, 1])  # [2, 256, 3]
        feat_offsets = ctr_offsets[..., 3:]
        new_features = feat_offsets
        ctr_offsets = ctr_offsets[..., :3]

        if self.max_offset_limit is not None:
            max_offset_limit = self.max_offset_limit.expand(xyz.shape)
            limited_ctr_offsets = paddle.where(ctr_offsets > max_offset_limit,
                                               max_offset_limit, ctr_offsets)
            min_offset_limit = -1 * max_offset_limit
            limited_ctr_offsets = paddle.where(
                limited_ctr_offsets < min_offset_limit, min_offset_limit,
                limited_ctr_offsets)
            vote_xyz = xyz + limited_ctr_offsets
        else:
            vote_xyz = xyz + ctr_offsets

        return vote_xyz, new_features, xyz, ctr_offsets
