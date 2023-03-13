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
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/roi_heads/pvrcnn_head.py
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""

import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.common import pointnet2_stack as pointnet2_stack_modules
from paddle3d.models.heads.roi_heads.roi_head_base import RoIHeadBase
from paddle3d.models.layers import (constant_init, kaiming_normal_init,
                                    xavier_normal_init)


@manager.HEADS.add_component
class PVRCNNHead(RoIHeadBase):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels,
            config=self.model_cfg["roi_grid_pool"])

        grid_size = self.model_cfg["roi_grid_pool"]["grid_size"]
        pre_channel = grid_size * grid_size * grid_size * num_c_out
        self.pre_channel = pre_channel

        shared_fc_list = []
        for k in range(0, self.model_cfg["shared_fc"].__len__()):
            shared_fc_list.extend([
                nn.Conv1D(
                    pre_channel,
                    self.model_cfg["shared_fc"][k],
                    kernel_size=1,
                    bias_attr=False),
                nn.BatchNorm1D(self.model_cfg["shared_fc"][k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg["shared_fc"][k]

            if k != self.model_cfg["shared_fc"].__len__(
            ) - 1 and self.model_cfg["dp_ratio"] > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg["dp_ratio"]))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.num_class,
            fc_list=self.model_cfg["cls_fc"])
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg["reg_fc"])
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init not in ['kaiming', 'xavier', 'normal']:
            raise NotImplementedError

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D) or isinstance(m, nn.Conv1D):
                if weight_init == 'normal':
                    m.weight.set_value(
                        paddle.normal(mean=0, std=0.001, shape=m.weight.shape))
                elif weight_init == 'kaiming':
                    kaiming_normal_init(
                        m.weight, reverse=isinstance(m, nn.Linear))
                elif weight_init == 'xavier':
                    xavier_normal_init(
                        m.weight, reverse=isinstance(m, nn.Linear))

                if m.bias is not None:
                    constant_init(m.bias, value=0)
            elif isinstance(m, nn.BatchNorm1D):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)
        self.reg_layers[-1].weight.set_value(
            paddle.normal(
                mean=0, std=0.001, shape=self.reg_layers[-1].weight.shape))

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict[
            'point_cls_scores'].reshape([-1, 1])

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg["roi_grid_pool"]
            ["grid_size"])  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.reshape(
            [batch_size, -1, 3])  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = paddle.zeros((batch_size, ), dtype='int32')
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum().astype(
                xyz_batch_cnt.dtype)

        new_xyz = global_roi_grid_points.reshape([-1, 3])
        new_xyz_batch_cnt = paddle.full((batch_size, ),
                                        global_roi_grid_points.shape[1],
                                        dtype='int32')
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz,
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features,
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.reshape([
            -1, self.model_cfg["roi_grid_pool"]["grid_size"]**3,
            pooled_features.shape[-1]
        ])  # (BxN, 6x6x6, C)
        return pooled_features

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict,
            nms_config=self.model_cfg["nms_config"]
            ['train' if self.training else 'test'])
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg["roi_grid_pool"]["grid_size"]
        pooled_features = pooled_features.transpose([0, 2, 1])

        shared_features = self.shared_fc_layer(
            pooled_features.reshape([-1, self.pre_channel, 1]))
        rcnn_cls = self.cls_layers(shared_features).transpose(
            [0, 2, 1]).squeeze(axis=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(
            [0, 2, 1]).squeeze(axis=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'],
                rois=batch_dict['rois'],
                cls_preds=rcnn_cls,
                box_preds=rcnn_reg)
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
