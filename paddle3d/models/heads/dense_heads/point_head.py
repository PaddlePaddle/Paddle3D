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
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/dense_heads/point_head_simple.py
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.layers import param_init
from paddle3d.models.losses import SigmoidFocalClassificationLoss
from paddle3d.ops import roiaware_pool3d
from paddle3d.models.common import enlarge_box3d


@manager.HEADS.add_component
class PointHeadSimple(nn.Layer):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """

    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class if not self.model_cfg['class_agnostic'] else 1

        self.build_losses(self.model_cfg['loss_config'])
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg['cls_fc'],
            input_channels=input_channels,
            output_channels=self.num_class)
        self.init_weights()

    def init_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, (nn.Linear)):
                param_init.reset_parameters(layer)
            if isinstance(layer, nn.BatchNorm1D):
                param_init.constant_init(layer.weight, value=1)
                param_init.constant_init(layer.bias, value=0)

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert len(
            gt_boxes.shape) == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert len(point_coords.shape) in [
            2
        ], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = enlarge_box3d(
            gt_boxes.reshape([-1, gt_boxes.shape[-1]]),
            extra_width=self.model_cfg['target_config']
            ['gt_extra_width']).reshape([batch_size, -1, gt_boxes.shape[-1]])
        targets_dict = self.assign_stack_targets(
            points=point_coords,
            gt_boxes=gt_boxes,
            extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True,
            use_ball_constraint=False,
            ret_part_labels=False)

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('use_point_features_before_fusion', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(
            point_features)  # (total_points, num_class)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }

        point_cls_scores = F.sigmoid(point_cls_preds)
        batch_dict['point_cls_scores'] = point_cls_scores.max(axis=-1)

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        self.forward_ret_dict = ret_dict

        return batch_dict

    def build_losses(self, losses_cfg):
        self.add_sublayer('cls_loss_func',
                          SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0))

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias_attr=False),
                nn.BatchNorm1D(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias_attr=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self,
                             points,
                             gt_boxes,
                             extend_gt_boxes=None,
                             ret_box_labels=False,
                             ret_part_labels=False,
                             set_ignore_flag=True,
                             use_ball_constraint=False,
                             central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(
            points.shape
        ) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape
                   ) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(
                       gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = paddle.zeros((points.shape[0], ), dtype='int64')
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = paddle.zeros((bs_mask.sum(), ),
                                                   dtype='int64')
            box_idxs_of_pts = roiaware_pool3d.points_in_boxes_gpu(
                points_single.unsqueeze(axis=0),
                gt_boxes[k:k + 1, :, 0:7]).astype('int64').squeeze(axis=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d.points_in_boxes_gpu(
                    points_single.unsqueeze(axis=0),
                    extend_gt_boxes[k:k + 1, :, 0:7]).astype('int64').squeeze(
                        axis=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = (paddle.linalg.norm(
                    (box_centers - points_single), axis=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[
                fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:,
                                                                             -1].astype(
                                                                                 'int64'
                                                                             )
            point_cls_labels[bs_mask] = point_cls_labels_single

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                raise NotImplementedError

            if ret_part_labels:
                raise NotImplementedError

        targets_dict = {
            'point_cls_labels': point_cls_labels,
        }
        return targets_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].reshape(
            [-1])
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].reshape(
            [-1, self.num_class])

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).astype('float32')
        pos_normalizer = positives.sum(axis=0).astype('float32')
        cls_weights /= paddle.clip(pos_normalizer, min=1.0)

        one_hot_targets = F.one_hot(
            point_cls_labels * (point_cls_labels >= 0).astype(
                point_cls_labels.dtype),
            num_classes=self.num_class + 1)
        one_hot_targets = one_hot_targets[:, 1:]
        one_hot_targets.stop_gradient = True
        cls_loss_src = self.cls_loss_func(
            point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg['loss_config']['loss_weights']
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls,
            'point_pos_num': pos_normalizer
        })
        return point_loss_cls, tb_dict
