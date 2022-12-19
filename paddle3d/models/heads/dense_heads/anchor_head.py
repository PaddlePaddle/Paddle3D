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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Normal

from paddle3d.apis import manager
from paddle3d.models.layers import reset_parameters
from paddle3d.models.losses import (SigmoidFocalClassificationLoss,
                                    WeightedCrossEntropyLoss,
                                    WeightedSmoothL1Loss)
from paddle3d.utils.box_coder import ResidualCoder

from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.axis_aligned_target_assigner import \
    AxisAlignedTargetAssigner

__all__ = ['AnchorHeadSingle']


@manager.HEADS.add_component
class AnchorHeadSingle(nn.Layer):
    def __init__(self, model_cfg, input_channels, class_names, voxel_size,
                 point_cloud_range, anchor_target_cfg,
                 predict_boxes_when_training, anchor_generator_cfg,
                 num_dir_bins, loss_weights):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = len(class_names)
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.anchor_generator_cfg = anchor_generator_cfg
        self.num_dir_bins = num_dir_bins
        self.loss_weights = loss_weights
        self.box_coder = ResidualCoder(num_dir_bins=num_dir_bins)
        point_cloud_range = np.asarray(point_cloud_range)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        self.anchors_list, self.num_anchors_per_location = self.generate_anchors(
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size)
        self.anchors = paddle.concat(self.anchors_list, axis=-3)
        # [x for x in anchors]
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2D(
            input_channels,
            self.num_anchors_per_location * self.num_class,
            kernel_size=1)

        self.conv_box = nn.Conv2D(
            input_channels,
            self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1)
        self.target_assigner = AxisAlignedTargetAssigner(
            anchor_generator_cfg,
            anchor_target_cfg,
            class_names=self.class_names,
            box_coder=self.box_coder)
        self.conv_dir_cls = nn.Conv2D(
            input_channels,
            self.num_anchors_per_location * num_dir_bins,
            kernel_size=1)
        self.forward_ret_dict = {}
        self.reg_loss_func = WeightedSmoothL1Loss(
            code_weights=loss_weights["code_weights"])
        self.cls_loss_func = SigmoidFocalClassificationLoss(
            alpha=0.25, gamma=2.0)
        self.dir_loss_func = WeightedCrossEntropyLoss()
        self.init_weight()

    def init_weight(self):

        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                reset_parameters(sublayer)
        bias_shape = self.conv_cls.bias.shape
        temp_value = paddle.ones(bias_shape) * -paddle.log(
            paddle.to_tensor((1.0 - 0.01) / 0.01))
        self.conv_cls.bias.set_value(temp_value)
        weight_shape = self.conv_box.weight.shape
        self.conv_box.weight.set_value(
            paddle.normal(mean=0.0, std=0.001, shape=weight_shape))

    def generate_anchors(self, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=self.anchor_generator_cfg)
        feature_map_size = [
            grid_size[:2] // config['feature_map_stride']
            for config in self.anchor_generator_cfg
        ]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(
            feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.zeros(
                    [*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = paddle.concat((anchors, pad_zeros), axis=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def generate_predicted_boxes(self,
                                 batch_size,
                                 cls_preds,
                                 box_preds,
                                 dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        # anchors = paddle.concat(self.anchors, axis=-3)
        anchors = self.anchors
        num_anchors = paddle.shape(
            anchors.reshape([-1, paddle.shape(anchors)[5]]))[0]
        batch_anchors = anchors.reshape([1, -1, paddle.shape(anchors)[5]]).tile(
            [batch_size, 1, 1])
        batch_cls_preds = cls_preds.reshape([batch_size, num_anchors, -1]) \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.reshape([batch_size, num_anchors, -1]) if not isinstance(box_preds, list) \
            else paddle.concat(box_preds, axis=1).reshape([batch_size, num_anchors, -1])
        batch_box_preds = self.box_coder.decode_paddle(batch_box_preds,
                                                       batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg['dir_offset']
            dir_limit_offset = self.model_cfg['dir_limit_offset']
            dir_cls_preds = dir_cls_preds.reshape([batch_size, num_anchors, -1]) if not isinstance(dir_cls_preds, list) \
                else paddle.concat(dir_cls_preds, axis=1).reshape([batch_size, num_anchors, -1])
            dir_labels = paddle.argmax(dir_cls_preds, axis=-1)

            period = (2 * np.pi / self.num_dir_bins)
            dir_rot = self.limit_period(batch_box_preds[..., 6] - dir_offset,
                                        dir_limit_offset, period)
            batch_box_preds[
                ..., 6] = dir_rot + dir_offset + period * dir_labels.cast(
                    batch_box_preds.dtype)

        return batch_cls_preds, batch_box_preds

    def limit_period(self, val, offset=0.5, period=np.pi):
        ans = val - paddle.floor(val / period + offset) * period
        return ans

    def forward(self, data_dict):

        spatial_features_2d = data_dict['spatial_features_2d']
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        cls_preds = cls_preds.transpose([0, 2, 3, 1])  # [N, H, W, C]
        box_preds = box_preds.transpose([0, 2, 3, 1])  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
        dir_cls_preds = dir_cls_preds.transpose([0, 2, 3, 1])
        self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds

        if self.training:
            targets_dict = self.target_assigner.assign_targets(
                self.anchors_list, data_dict['gt_boxes'])
            self.forward_ret_dict.update(targets_dict)
        if not self.training or self.predict_boxes_when_training:
            if getattr(self, 'in_export_mode', False):
                batch_size = 1
            else:
                batch_size = data_dict['batch_size']
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_size,
                cls_preds=cls_preds,
                box_preds=box_preds,
                dir_cls_preds=dir_cls_preds)
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives)
        reg_weights = positives
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True)
        reg_weights /= paddle.clip(pos_normalizer, min=1.0)
        cls_weights /= paddle.clip(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.cast(box_cls_labels.dtype)

        one_hot_targets = []
        for b in range(batch_size):
            one_hot_targets.append(
                F.one_hot(cls_targets[b], num_classes=self.num_class + 1))
        one_hot_targets = paddle.stack(one_hot_targets)
        cls_preds = cls_preds.reshape([batch_size, -1, self.num_class])
        one_hot_targets = one_hot_targets[..., 1:]
        one_hot_targets.stop_gradient = True
        cls_loss_src = self.cls_loss_func(
            cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.loss_weights['cls_weight']
        tb_dict = {'rpn_loss_cls': cls_loss.item()}

        return cls_loss, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.cast("float32")
        pos_normalizer = positives.sum(1, keepdim=True)
        reg_weights /= paddle.clip(pos_normalizer, min=1.0)

        anchors = self.anchors

        anchors = anchors.reshape([1, -1,
                                   anchors.shape[-1]]).tile([batch_size, 1, 1])
        box_preds = box_preds.reshape([
            batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location
        ])
        box_preds_sin, reg_targets_sin = self.add_sin_difference(
            box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, \
                                        weights=reg_weights)  # [N, M]

        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.loss_weights['loc_weight']
        box_loss = loc_loss
        tb_dict = {'rpn_loss_loc': loc_loss.item()}

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors,
                box_reg_targets,
                dir_offset=self.model_cfg['dir_offset'],
                num_bins=self.num_dir_bins)

            dir_logits = box_dir_cls_preds.reshape(
                [batch_size, -1, self.num_dir_bins])
            weights = positives.cast("float32")
            weights /= paddle.clip(weights.sum(-1, keepdim=True), min=1.0)
            dir_targets.stop_gradient = True
            dir_loss = self.dir_loss_func(
                dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.loss_weights['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()
        return box_loss, tb_dict

    def add_sin_difference(self, boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = paddle.sin(boxes1[..., dim:dim + 1]) * paddle.cos(
            boxes2[..., dim:dim + 1])
        rad_tg_encoding = paddle.cos(boxes1[..., dim:dim + 1]) * paddle.sin(
            boxes2[..., dim:dim + 1])
        boxes1 = paddle.concat(
            [boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]],
            axis=-1)
        boxes2 = paddle.concat(
            [boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]],
            axis=-1)
        return boxes1, boxes2

    def get_direction_target(self,
                             anchors,
                             reg_targets,
                             one_hot=True,
                             dir_offset=0,
                             num_bins=2):
        batch_size = reg_targets.shape[0]

        anchors = anchors.reshape([batch_size, -1, anchors.shape[-1]])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = self.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = paddle.floor(
            offset_rot / (2 * np.pi / num_bins)).cast("int64")
        dir_cls_targets = paddle.clip(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = []
            for b in range(batch_size):
                dir_targets.append(
                    F.one_hot(dir_cls_targets[b], num_classes=num_bins))
            dir_cls_targets = paddle.stack(dir_targets)
        return dir_cls_targets
