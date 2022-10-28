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
"""
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/pfe/voxel_set_abstraction.py
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""

import numpy as np
import paddle
import paddle.nn as nn
from paddle import sparse

from paddle3d.apis import manager
from paddle3d.models.common import get_voxel_centers
from paddle3d.models.common import pointnet2_stack as pointnet2_stack_modules
from paddle3d.models.layers import param_init
from paddle3d.ops import pointnet2_ops


def bilinear_interpolate_paddle(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = paddle.floor(x).astype('int64')
    x1 = x0 + 1

    y0 = paddle.floor(y).astype('int64')
    y1 = y0 + 1

    x0 = paddle.clip(x0, 0, im.shape[1] - 1)
    x1 = paddle.clip(x1, 0, im.shape[1] - 1)
    y0 = paddle.clip(y0, 0, im.shape[0] - 1)
    y1 = paddle.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.astype(x.dtype) - x) * (y1.astype(y.dtype) - y)
    wb = (x1.astype(x.dtype) - x) * (y - y0.astype(y.dtype))
    wc = (x - x0.astype(x.dtype)) * (y1.astype(y.dtype) - y)
    wd = (x - x0.astype(x.dtype)) * (y - y0.astype(y.dtype))
    ans = paddle.transpose(
        (paddle.transpose(Ia, [1, 0]) * wa), [1, 0]) + paddle.transpose(
            paddle.transpose(Ib, [1, 0]) * wb, [1, 0]) + paddle.transpose(
                paddle.transpose(Ic, [1, 0]) * wc, [1, 0]) + paddle.transpose(
                    paddle.transpose(Id, [1, 0]) * wd, [1, 0])
    return ans


def sample_points_with_roi(rois,
                           points,
                           sample_radius_with_roi,
                           num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        distance = paddle.linalg.norm(
            points[:, None, :] - rois[None, :, 0:3], axis=-1)
        min_dis, min_dis_roi_idx = distance.min(axis=-1)
        roi_max_axis = paddle.linalg.norm(
            rois[min_dis_roi_idx, 3:6] / 2, axis=-1)
        point_mask = min_dis < roi_max_axis + sample_radius_with_roi
    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (
                points[start_idx:start_idx + num_max_points_of_part, None, :] -
                rois[None, :, 0:3]).norm(axis=-1)
            min_dis, min_dis_roi_idx = distance.min(axis=-1)
            roi_max_axis = paddle.linalg.norm(
                rois[min_dis_roi_idx, 3:6] / 2, axis=-1)
            cur_point_mask = min_dis < roi_max_axis + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = paddle.concat(point_mask_list, axis=0)

    sampled_points = points[:1] if point_mask.sum(
    ) == 0 else points[point_mask, :]

    return sampled_points, point_mask


@manager.POINT_ENCODERS.add_component
class VoxelSetAbstraction(nn.Layer):
    def __init__(self,
                 model_cfg,
                 voxel_size,
                 point_cloud_range,
                 num_bev_features=None,
                 num_rawpoint_features=None,
                 **kwargs):
        super(VoxelSetAbstraction, self).__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        sa_cfg = self.model_cfg["sa_layer"]

        self.sa_layers = nn.LayerList()
        self.sa_layer_names = []
        self.downsample_stride_map = {}
        c_in = 0
        for src_name in self.model_cfg["features_source"]:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_stride_map[src_name] = sa_cfg[src_name][
                "downsample_stride"]

            if sa_cfg[src_name].get('in_channels', None) is None:
                input_channels = sa_cfg[src_name]["mlps"][0][0] \
                    if isinstance(sa_cfg[src_name]["mlps"][0], list) else sa_cfg[src_name]["mlps"][0]
            else:
                input_channels = sa_cfg[src_name]['in_channels']

            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=input_channels, config=sa_cfg[src_name])
            self.sa_layers.append(cur_layer)
            self.sa_layer_names.append(src_name)

            c_in += cur_num_c_out

        if 'bev' in self.model_cfg["features_source"]:
            c_bev = num_bev_features
            c_in += c_bev

        self.num_rawpoint_features = num_rawpoint_features
        if 'raw_points' in self.model_cfg["features_source"]:
            self.sa_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=num_rawpoint_features - 3,
                config=sa_cfg['raw_points'])

            c_in += cur_num_c_out

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg["out_channels"], bias_attr=False),
            nn.BatchNorm1D(self.model_cfg["out_channels"]),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg["out_channels"]
        self.num_point_features_before_fusion = c_in
        self.init_weights()

    def init_weights(self):
        for layer in self.vsa_point_feature_fusion.sublayers():
            if isinstance(layer, (nn.Linear)):
                param_init.reset_parameters(layer)
            if isinstance(layer, nn.BatchNorm1D):
                param_init.constant_init(layer.weight, value=1)
                param_init.constant_init(layer.bias, value=0)

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size,
                                      bev_stride):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (
            keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (
            keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            bs_mask = (keypoints[:, 0] == k)

            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_bev_features = bev_features[k].transpose((1, 2, 0))  # (H, W, C)
            point_bev_features = bilinear_interpolate_paddle(
                cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = paddle.concat(
            point_bev_features_list, axis=0)  # (N1 + N2 + ..., C)
        return point_bev_features

    def get_sampled_points(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 ind.concates [bs_idx, x, y, z]
        """
        batch_size = batch_dict['batch_size']
        if self.model_cfg["point_source"] == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].astype('int64')
        elif self.model_cfg["point_source"] == 'voxel_centers':
            raise NotImplementedError
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(axis=0)  # (1, N, 3)
            if self.model_cfg["sample_method"] == 'FPS':
                cur_pt_idxs = pointnet2_ops.farthest_point_sample(
                    sampled_points[:, :, 0:3],
                    self.model_cfg["num_keypoints"]).astype('int64')

                if sampled_points.shape[1] < self.model_cfg["num_keypoints"]:
                    times = int(self.model_cfg["num_keypoints"] /
                                sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.tile(
                        [1, times])[:self.model_cfg["num_keypoints"]]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(axis=0)

            elif self.model_cfg["sample_method"] == 'SPC':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = paddle.concat(
            keypoints_list, axis=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
        if len(keypoints.shape) == 3:
            batch_idx = paddle.arange(batch_size).reshape([-1, 1]).tile(
                [1, keypoints.shape[1]]).reshape([-1, 1])
            keypoints = paddle.concat(
                (batch_idx.astype('float32'), keypoints.reshape([-1, 3])),
                axis=1)

        return keypoints

    @staticmethod
    def aggregate_keypoint_features_from_one_source(
            batch_size,
            aggregate_func,
            xyz,
            xyz_features,
            xyz_bs_idxs,
            new_xyz,
            new_xyz_batch_cnt,
            filter_neighbors_with_roi=False,
            radius_of_neighbor=None,
            num_max_points_of_part=200000,
            rois=None):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        xyz_batch_cnt = paddle.zeros((batch_size, ), dtype='int32')
        if filter_neighbors_with_roi:
            point_features = paddle.concat(
                (xyz,
                 xyz_features), axis=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = (xyz_bs_idxs == bs_idx)
                _, valid_mask = sample_points_with_roi(
                    rois=rois[bs_idx],
                    points=xyz[bs_mask],
                    sample_radius_with_roi=radius_of_neighbor,
                    num_max_points_of_part=num_max_points_of_part,
                )
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum().astype(
                    xyz_batch_cnt.dtype)

            valid_point_features = paddle.concat(point_features_list, axis=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:,
                                                3:] if xyz_features is not None else None
        else:
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum().astype(
                    xyz_batch_cnt.dtype)

        pooled_points, pooled_features = aggregate_func(
            xyz=xyz,
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=xyz_features,
        )
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict)

        point_features_list = []
        if 'bev' in self.model_cfg["features_source"]:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints,
                batch_dict['spatial_features'],
                batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride'])
            point_features_list.append(point_bev_features)

        batch_size = batch_dict['batch_size']

        new_xyz = keypoints[:, 1:4]
        new_xyz_batch_cnt = paddle.zeros((batch_size, ), dtype='int32')
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum().astype(
                new_xyz_batch_cnt.dtype)

        if 'raw_points' in self.model_cfg["features_source"]:
            raw_points = batch_dict['points']

            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size,
                aggregate_func=self.sa_rawpoints,
                xyz=raw_points[:, 1:4],
                xyz_features=raw_points[:, 4:]
                if self.num_rawpoint_features > 3 else None,
                xyz_bs_idxs=raw_points[:, 0],
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg["sa_layer"]
                ['raw_points'].get('filter_neighbor_with_roi', False),
                radius_of_neighbor=self.model_cfg["sa_layer"]['raw_points'].get(
                    'radius_of_neighbor_with_roi', None),
                rois=batch_dict.get('rois', None))
            point_features_list.append(pooled_features)

        for k, src_name in enumerate(self.sa_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][
                src_name].indices().transpose([1, 0])
            cur_features = batch_dict['multi_scale_3d_features'][
                src_name].values()

            xyz = get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_strides=self.downsample_stride_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range)
            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size,
                aggregate_func=self.sa_layers[k],
                xyz=xyz,
                xyz_features=cur_features,
                xyz_bs_idxs=cur_coords[:, 0],
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg["sa_layer"]
                [src_name].get('filter_neighbor_with_roi', False),
                radius_of_neighbor=self.model_cfg["sa_layer"][src_name].get(
                    'radius_of_neighbor_with_roi', None),
                rois=batch_dict.get('rois', None))
            point_features_list.append(pooled_features)

        point_features = paddle.concat(point_features_list, axis=-1)

        batch_dict['point_features_before_fusion'] = point_features.reshape(
            (-1, point_features.shape[-1]))
        point_features = self.vsa_point_feature_fusion(
            point_features.reshape((-1, point_features.shape[-1])))

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = keypoints  # (BxN, 4)
        return batch_dict
