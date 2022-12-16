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
This code is based on https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/readers/pillar_encoder.py
Ths copyright of tianweiy/CenterPoint is as follows:
MIT License [see LICENSE for details].

https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/readers/pillar_encoder.py fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Uniform

from paddle3d.apis import manager

from .voxel_encoder import get_paddings_indicator

__all__ = ['PillarFeatureNet', 'build_linear_layer', 'build_norm_layer']


def build_linear_layer(in_channels, out_channels, bias=True):
    """Build linear layer."""
    bound = 1 / math.sqrt(in_channels)
    param_attr = ParamAttr(initializer=Uniform(-bound, bound))
    bias_attr = False
    if bias:
        bias_attr = ParamAttr(initializer=Uniform(-bound, bound))
    return nn.Linear(
        in_channels, out_channels, weight_attr=param_attr, bias_attr=bias_attr)


def build_norm_layer(cfg, num_features, weight_attr=True, bias_attr=True):
    """Build normalization layer."""
    norm_layer = getattr(nn, cfg['type'])(
        num_features,
        momentum=1 - cfg['momentum'],
        epsilon=cfg['eps'],
        weight_attr=ParamAttr(initializer=Constant(
            value=1)) if weight_attr else False,
        bias_attr=ParamAttr(initializer=Constant(
            value=0)) if bias_attr else False)

    return norm_layer


class PFNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 max_num_points_in_voxel=20,
                 norm_cfg=dict(type='BatchNorm1D', eps=1e-3, momentum=0.01),
                 last_layer=False):
        super(PFNLayer, self).__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.norm = build_norm_layer(norm_cfg, self.units)
        self.linear = build_linear_layer(in_channels, self.units, bias=False)
        self.max_num_points_in_voxel = max_num_points_in_voxel

    def forward(self, inputs, num_voxels=None):
        x = self.linear(inputs)
        x = self.norm(x.transpose(perm=[0, 2, 1])).transpose(perm=[0, 2, 1])
        x = F.relu(x)

        # x_max = paddle.max(x, axis=1, keepdim=True)
        # TODO(luoqianhui): remove the following complicated max operation
        # paddle.max mistakenly backwards gradient to all elements when they are same,
        # to align with paddle implement, we recombine paddle apis to backwards gradient
        # to the last one.
        # Note: when the input elements are same, paddle max and argmax treat the last one
        # as the maximum value, but paddle argmax, numpy max and argmax treat the first one.
        max_idx = paddle.argmax(x, axis=1)
        data = x.transpose([0, 2,
                            1]).reshape([-1, self.max_num_points_in_voxel])
        index = max_idx.reshape([-1, 1])
        sample = paddle.index_sample(data, index)
        x_max = sample.reshape([-1, self.units, 1]).transpose([0, 2, 1])

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.tile([1, self.max_num_points_in_voxel, 1])
            x_concatenated = paddle.concat([x, x_repeat], axis=2)
            return x_concatenated


@manager.VOXEL_ENCODERS.add_component
class PillarFeatureNet(nn.Layer):
    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 max_num_points_in_voxel=20,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 legacy=True):
        super(PillarFeatureNet, self).__init__()
        self.legacy = legacy
        self.in_channels = in_channels
        # with cluster center
        in_channels += 3
        # with voxel center
        in_channels += 2
        if with_distance:
            in_channels += 1
        self.with_distance = with_distance
        # Create PillarFeatureNet layers
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        norm_cfg = dict(type='BatchNorm1D', eps=1e-3, momentum=0.01)
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    max_num_points_in_voxel=max_num_points_in_voxel,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer))
        self.pfn_layers = nn.LayerList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range
        self.max_num_points_in_voxel = max_num_points_in_voxel

    def forward(self, features, num_points_per_voxel, coors):
        """Forward function.

        Args:
            features (paddle.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points_per_voxel (paddle.Tensor): Number of points in each pillar.
            coors (paddle.Tensor): Coordinates of each voxel.

        Returns:
            paddle.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        features_sum = paddle.sum(features[:, :, :3], axis=1, keepdim=True)
        points_mean = features_sum / paddle.cast(
            num_points_per_voxel, features.dtype).reshape([-1, 1, 1])
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if not self.legacy:
            f_center = paddle.zeros_like(features[:, :, :2])
            f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].reshape(
                [-1, 1]).astype(dtype) * self.vx + self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].reshape(
                [-1, 1]).astype(dtype) * self.vy + self.y_offset)

        else:
            f_center = features[:, :, :2]
            f_center[:, :, 0] = f_center[:, :, 0] - (coors[:, 3].reshape(
                [-1, 1]).astype(features.dtype) * self.vx + self.x_offset)
            f_center[:, :, 1] = f_center[:, :, 1] - (coors[:, 2].reshape(
                [-1, 1]).astype(features.dtype) * self.vy + self.y_offset)

        features_ls.append(f_center)

        if self.with_distance:
            points_dist = paddle.linalg.norm(
                features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        features = paddle.concat(features_ls, axis=-1)
        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        mask = get_paddings_indicator(num_points_per_voxel,
                                      self.max_num_points_in_voxel)
        mask = paddle.reshape(
            mask, [-1, self.max_num_points_in_voxel, 1]).astype(features.dtype)
        features = features * mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points_per_voxel)
        return features.squeeze()
