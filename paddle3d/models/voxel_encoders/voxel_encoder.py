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
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.layers.param_init import constant_init, reset_parameters

__all__ = ['VoxelMean', 'HardVFE']


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


class VFELayer(nn.Layer):
    """Voxel Feature Encoder layer.

    The voxel encoder is composed of a series of these layers.
    This module do not support average pooling and only support to use
    max pooling to gather features inside a VFE.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        max_out (bool): Whether aggregate the features of points inside
            each voxel and only return voxel features.
        cat_max (bool): Whether concatenate the aggregated features
            and pointwise features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BatchNorm1D', eps=1e-3, momentum=1 - 0.01),
                 max_out=True,
                 cat_max=True):
        super(VFELayer, self).__init__()
        self.cat_max = cat_max
        self.max_out = max_out
        self.out_channels = out_channels
        # self.units = int(out_channels / 2)

        self.norm = nn.BatchNorm1D(
            out_channels, epsilon=1e-3, momentum=1 - 0.01)
        self.linear = nn.Linear(in_channels, out_channels, bias_attr=False)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (paddle.Tensor): Voxels features of shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.

        Returns:
            paddle.Tensor: Voxel features. There are three mode under which the
                features have different meaning.
                - `max_out=False`: Return point-wise features in
                    shape (N, M, C).
                - `max_out=True` and `cat_max=False`: Return aggregated
                    voxel features in shape (N, C)
                - `max_out=True` and `cat_max=True`: Return concatenated
                    point-wise features in shape (N, M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]

        x = self.linear(inputs)
        x = self.norm(x.transpose([0, 2, 1])).transpose([0, 2, 1])

        pointwise = F.relu(x)
        # [K, T, units]
        if self.max_out:
            # aggregated = paddle.max(pointwise, axis=1, keepdim=True)
            max_idx = paddle.argmax(pointwise, axis=1)
            data = pointwise.transpose([0, 2, 1]).reshape([-1, voxel_count])
            index = max_idx.reshape([-1, 1])
            sample = paddle.index_sample(data, index)
            aggregated = sample.reshape([-1, self.out_channels,
                                         1]).transpose([0, 2, 1])
        else:
            # this is for fusion layer
            return pointwise

        if not self.cat_max:
            return aggregated.squeeze(1)
        else:
            # [K, 1, units]
            repeated = aggregated.tile([1, voxel_count, 1])
            concatenated = paddle.concat([pointwise, repeated], axis=2)
            # [K, T, 2 * units]
            return concatenated


@manager.VOXEL_ENCODERS.add_component
class HardVFE(nn.Layer):
    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BatchNorm1D', eps=1e-3, momentum=1 - 0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False):
        super(HardVFE, self).__init__()
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                VFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max))
            self.vfe_layers = nn.LayerList(vfe_layers)
        self.num_vfe = len(vfe_layers)

        self.fusion_layer = None
        if fusion_layer is not None:
            raise NotImplementedError

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, (nn.Conv1D, nn.Conv2D)):
                reset_parameters(m)
            elif isinstance(m, nn.Linear):
                reset_parameters(m, reverse=True)
            elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D)):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)

        self.apply(_init_weights)

    def forward(self,
                features,
                num_points,
                coors,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (paddle.Tensor): Features of voxels, shape is MxNxC.
            num_points (paddle.Tensor): Number of points in each voxel.
            coors (paddle.Tensor): Coordinates of voxels, shape is Mx(1+NDim).
            img_feats (list[paddle.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = (
                features[:, :, :3].sum(axis=1, keepdim=True) /
                num_points.astype(features.dtype).reshape([-1, 1, 1]))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = paddle.zeros([features.shape[0], features.shape[1], 3],
                                    dtype=features.dtype)
            f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].astype(
                features.dtype).unsqueeze(1) * self.vx + self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].astype(
                features.dtype).unsqueeze(1) * self.vy + self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (coors[:, 1].astype(
                features.dtype).unsqueeze(1) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = paddle.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        voxel_feats = paddle.concat(features_ls, axis=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty.
        # Need to ensure that empty voxels remain set to zeros.
        voxel_count = voxel_feats.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count)
        voxel_feats *= mask.unsqueeze(-1).astype(voxel_feats.dtype)

        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)

        if (self.fusion_layer is not None and img_feats is not None):
            raise NotImplementedError

        return voxel_feats
