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
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/roi_heads/roi_head_template.py
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.models.common import class_agnostic_nms, rotate_points_along_z
from paddle3d.models.heads.roi_heads.target_assigner.proposal_target_layer import \
    ProposalTargetLayer
from paddle3d.models.losses import WeightedSmoothL1Loss, get_corner_loss_lidar
from paddle3d.utils import box_coder as box_coder_utils


class RoIHeadBase(nn.Layer):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(
            box_coder_utils, self.model_cfg["target_config"]["box_coder"])(
                **self.model_cfg["target_config"].get('box_coder_config', {}))
        self.proposal_target_layer = ProposalTargetLayer(
            roi_sampler_cfg=self.model_cfg["target_config"])
        self.build_losses(self.model_cfg["loss_config"])
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_sublayer(
            'reg_loss_func',
            WeightedSmoothL1Loss(
                code_weights=losses_cfg["loss_weights"]['code_weights']))

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1D(
                    pre_channel, fc_list[k], kernel_size=1, bias_attr=False),
                nn.BatchNorm1D(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg["dp_ratio"] >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg["dp_ratio"]))
        fc_layers.append(
            nn.Conv1D(
                pre_channel, output_channels, kernel_size=1, bias_attr=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @paddle.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict

        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = paddle.zeros((batch_size, nms_config["nms_post_maxsize"],
                             batch_box_preds.shape[-1]))
        roi_scores = paddle.zeros((batch_size, nms_config["nms_post_maxsize"]))
        roi_labels = paddle.zeros((batch_size, nms_config["nms_post_maxsize"]),
                                  dtype='int64')

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores = paddle.max(cls_preds, axis=1)
            cur_roi_labels = paddle.argmax(cls_preds, axis=1)

            if nms_config['multi_class_nms']:
                raise NotImplementedError
            else:
                selected_score, selected_label, selected_box = class_agnostic_nms(
                    box_scores=cur_roi_scores,
                    box_preds=box_preds,
                    label_preds=cur_roi_labels,
                    nms_config=nms_config)

            rois[index, :selected_label.shape[0], :] = selected_box
            roi_scores[index, :selected_label.shape[0]] = selected_score
            roi_labels[index, :selected_label.shape[0]] = selected_label

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with paddle.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        #index = paddle.to_tensor([0, 1, 2], dtype='int32')
        #selected_rois = paddle.index_select(gt_of_rois, index=index, axis=-1)
        #gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        selected_rois = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 0:3] = selected_rois
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = rotate_points_along_z(
            points=gt_of_rois.reshape([-1, 1, gt_of_rois.shape[-1]]),
            angle=-roi_ry.reshape([-1])).reshape(
                [batch_size, -1, gt_of_rois.shape[-1]])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label <
                                                         np.pi * 1.5)
        if opposite_flag.numel() > 0:
            heading_label[opposite_flag] = (
                heading_label[opposite_flag] + np.pi) % (
                    2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        if flag.numel() > 0:
            heading_label[
                flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = paddle.clip(
            heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg["loss_config"]
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].reshape([-1])
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:
                                                            code_size].reshape(
                                                                [-1, code_size])
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.reshape([-1, code_size]).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.astype('int32').sum()

        tb_dict = {}

        if loss_cfgs['reg_loss'] == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().reshape([-1, code_size])
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_paddle(
                gt_boxes3d_ct.reshape([rcnn_batch_size, code_size]),
                rois_anchor)

            #reg_targets.stop_gradient = True
            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.reshape([rcnn_batch_size, -1]).unsqueeze(axis=0),
                reg_targets.unsqueeze(axis=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.reshape([rcnn_batch_size, -1]) *
                             fg_mask.unsqueeze(axis=-1).astype('float32')
                             ).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs['loss_weights'][
                'rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg

            if loss_cfgs["corner_loss_regularization"] and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.reshape([rcnn_batch_size, -1])[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.reshape([-1, code_size])[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.reshape([1, -1, code_size])
                batch_anchors = fg_roi_boxes3d.clone()
                roi_ry = fg_roi_boxes3d[:, :, 6].reshape([-1])
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].reshape([-1, 3])
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_paddle(
                    fg_rcnn_reg.reshape([batch_anchors.shape[0], -1,
                                         code_size]),
                    batch_anchors).reshape([-1, code_size])

                rcnn_boxes3d = rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(axis=1), roi_ry).squeeze(axis=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                #gt_of_rois_src.stop_gradient = True
                loss_corner = get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7], gt_of_rois_src[fg_mask][:, 0:7])
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs["loss_weights"][
                    'rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg["loss_config"]
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].reshape([-1])
        if loss_cfgs['cls_loss'] == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.reshape([-1])
            #rcnn_cls_labels.stop_gradient = True
            batch_loss_cls = F.binary_cross_entropy(
                F.sigmoid(rcnn_cls_flat),
                rcnn_cls_labels.astype('float32'),
                reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).astype('float32')
            rcnn_loss_cls = (
                batch_loss_cls * cls_valid_mask).sum() / paddle.clip(
                    cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs['cls_loss'] == 'CrossEntropy':
            #rcnn_cls_labels.stop_gradient = True
            batch_loss_cls = F.cross_entropy(
                rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).astype('float32')
            rcnn_loss_cls = (
                batch_loss_cls * cls_valid_mask).sum() / paddle.clip(
                    cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs["loss_weights"][
            'rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(
            self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(
            self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.reshape(
            [batch_size, -1, cls_preds.shape[-1]])
        batch_box_preds = box_preds.reshape([batch_size, -1, code_size])

        roi_ry = rois[:, :, 6].reshape([-1])
        roi_xyz = rois[:, :, 0:3].reshape([-1, 3])
        local_rois = rois.clone()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_paddle(
            batch_box_preds, local_rois).reshape([-1, code_size])

        batch_box_preds = rotate_points_along_z(
            batch_box_preds.unsqueeze(axis=1), roi_ry).squeeze(axis=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.reshape([batch_size, -1, code_size])
        return batch_cls_preds, batch_box_preds

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.reshape([-1, rois.shape[-1]])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(
            rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]).squeeze(axis=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(axis=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = paddle.ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.tile([batch_size_rcnn, 1,
                                    1]).astype('float32')  # (B, 6x6x6, 3)

        local_roi_size = rois.reshape([batch_size_rcnn, -1])[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(axis=1) \
                          - (local_roi_size.unsqueeze(axis=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points
