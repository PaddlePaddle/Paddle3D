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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager

__all__ = [
    "PointPillarsLoss", "SigmoidFocalClassificationLoss",
    "WeightedSmoothL1RegressionLoss", "WeightedSoftmaxClassificationLoss"
]


@manager.LOSSES.add_component
class PointPillarsLoss(nn.Layer):
    def __init__(self,
                 num_classes,
                 classification_loss,
                 regression_loss,
                 direction_loss=None,
                 classification_loss_weight=1.0,
                 regression_loss_weight=2.0,
                 direction_loss_weight=1.0,
                 fg_cls_weight=1.0,
                 bg_cls_weight=1.0,
                 encode_rot_error_by_sin=True,
                 use_direction_classifier=True,
                 encode_background_as_zeros=True,
                 box_code_size=7):
        super(PointPillarsLoss, self).__init__()

        self.num_classes = num_classes

        self.cls_loss = classification_loss
        self.cls_loss_w = classification_loss_weight

        self.reg_loss = regression_loss
        self.reg_loss_w = regression_loss_weight

        self.dir_loss = direction_loss
        self.dir_loss_w = direction_loss_weight

        self.fg_cls_weight = fg_cls_weight
        self.bg_cls_weight = bg_cls_weight

        self.encode_rot_error_by_sin = encode_rot_error_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.encode_background_as_zeros = encode_background_as_zeros
        self.box_code_size = box_code_size

    def compute_loss_weights(self, labels):
        """get cls_weights and reg_weights from labels.
        """
        cared = labels >= 0
        foregrounds = (labels > 0).astype('float32')
        backgrounds = (labels == 0).astype("float32")
        cls_weights = self.bg_cls_weight * backgrounds + self.fg_cls_weight * foregrounds
        reg_weights = foregrounds

        fg_normalizer = foregrounds.sum(1, keepdim=True)
        reg_weights /= paddle.clip(fg_normalizer, min=1.0)
        cls_weights /= paddle.clip(fg_normalizer, min=1.0)

        return cls_weights, reg_weights, cared

    def compute_cls_loss(self, cls_preds, cls_targets, cls_weights):
        cls_targets_onehot = F.one_hot(cls_targets, self.num_classes + 1)
        if self.encode_background_as_zeros:
            cls_targets_onehot = cls_targets_onehot[..., 1:]

        cls_loss = self.cls_loss(
            cls_preds, cls_targets_onehot, weights=cls_weights)  # [N, M]

        return cls_loss

    def compute_reg_loss(self, box_preds, reg_targets, reg_weights):
        if self.encode_rot_error_by_sin:
            # sin(pred - target) = sin(pred)cos(target) - cos(pred)sin(target)
            rad_pred_encoding = paddle.sin(box_preds[..., -1:]) * paddle.cos(
                reg_targets[..., -1:])
            rad_target_encoding = paddle.cos(box_preds[..., -1:]) * paddle.sin(
                reg_targets[..., -1:])
            box_preds = paddle.concat([box_preds[..., :-1], rad_pred_encoding],
                                      axis=-1)
            reg_targets = paddle.concat(
                [reg_targets[..., :-1], rad_target_encoding], axis=-1)

        reg_loss = self.reg_loss(
            box_preds, reg_targets, weights=reg_weights)  # [N, M]

        return reg_loss

    def compute_fg_bg_loss(self, cls_loss, labels):
        # cls_loss: [N, num_anchors, num_class]
        # labels: [N, num_anchors]
        batch_size = cls_loss.shape[0]
        if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
            cls_fg_loss = (labels > 0).astype(
                cls_loss.dtype) * cls_loss.reshape((batch_size, -1))
            cls_bg_loss = (labels == 0).astype(
                cls_loss.type) * cls_loss.reshape((batch_size, -1))
            cls_fg_loss = cls_fg_loss.sum() / batch_size
            cls_bg_loss = cls_bg_loss.sum() / batch_size
        else:
            cls_fg_loss = cls_loss[..., 1:].sum() / batch_size
            cls_bg_loss = cls_loss[..., 0].sum() / batch_size

        cls_fg_loss /= self.fg_cls_weight
        cls_bg_loss /= self.bg_cls_weight

        return cls_fg_loss, cls_bg_loss

    def compute_dir_cls_loss(self, dir_preds, reg_targets, labels, anchors):
        # batch_size = dir_preds.shape[0]
        # anchors = paddle.broadcast_to(anchors, [batch_size] + anchors.shape)
        rot_gt = reg_targets[..., -1] + anchors[..., -1]
        dir_targets = (rot_gt > 0).astype("int32")

        weights = (labels > 0).astype(dir_preds.dtype)
        weights /= paddle.clip(weights.sum(-1, keepdim=True), min=1.0)
        dir_loss = self.dir_loss(dir_preds, dir_targets, weights=weights)

        return dir_loss

    def forward(self,
                box_preds,
                cls_preds,
                reg_targets,
                labels,
                dir_preds=None,
                anchors=None):
        cls_weights, reg_weights, cared = self.compute_loss_weights(labels)
        cls_targets = labels * cared.astype(labels.dtype)

        cls_loss = self.compute_cls_loss(cls_preds, cls_targets, cls_weights)
        reg_loss = self.compute_reg_loss(box_preds, reg_targets, reg_weights)
        # cls_fg_loss, cls_bg_loss = self.compute_fg_bg_loss(cls_loss, labels)

        batch_size = box_preds.shape[0]
        total_loss = self.reg_loss_w * (reg_loss.sum(
        ) / batch_size) + self.cls_loss_w * (cls_loss.sum() / batch_size)

        loss_dict = dict(loss=total_loss)

        if self.use_direction_classifier:
            dir_loss = self.compute_dir_cls_loss(dir_preds, reg_targets, labels,
                                                 anchors)
            total_loss += self.dir_loss_w * (dir_loss.sum() / batch_size)
            loss_dict.update(dict(loss=total_loss))

        return loss_dict


@manager.LOSSES.add_component
class SigmoidFocalClassificationLoss(nn.Layer):
    """
    Sigmoid focal cross entropy loss.

    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.

    Args:
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, prediction, target, weights, class_indices=None):
        """
        Compute loss function.

        Args:
            prediction: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing the predicted logits for each class
            target: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
                If provided, computes loss only for the specified class indices.

        Returns:
            loss: a float tensor of shape [batch_size, num_anchors, num_classes]
                representing the value of the loss function.
        """
        weights = weights.unsqueeze(2)
        if class_indices is not None:
            mask = paddle.zeros((prediction.shape[2], ), dtype=prediction.dtype)
            mask[class_indices] = 1.
            weights *= mask.reshape((1, 1, -1))

        per_entry_cross_entropy = paddle.clip(
            prediction, min=0) - prediction * target.astype(
                prediction.dtype) + paddle.log1p(
                    paddle.exp(-paddle.abs(prediction)))

        pred_prob = F.sigmoid(prediction)
        p_t = (target * pred_prob) + ((1 - target) * (1 - pred_prob))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = paddle.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target * self._alpha + (1 - target) * (
                1 - self._alpha)

        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor * per_entry_cross_entropy

        return focal_cross_entropy_loss * weights


@manager.LOSSES.add_component
class WeightedSmoothL1RegressionLoss(nn.Layer):
    """
    Smooth L1 regression loss function.

    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, code_weights=None, codewise=True):
        super(WeightedSmoothL1RegressionLoss, self).__init__()
        self._sigma = sigma
        if code_weights is not None:
            self._code_weights = paddle.to_tensor(
                code_weights, dtype="float32").reshape((1, 1, -1))
        else:
            self._code_weights = None
        self._codewise = codewise

    def forward(self, prediction, target, weights=None):
        """Compute loss function.

        Args:
            prediction: A float tensor of shape [batch_size, num_anchors,
                code_size] representing the (encoded) predicted locations of objects.
            target: A float tensor of shape [batch_size, num_anchors,
                code_size] representing the regression targets
            weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
            loss: a float tensor of shape [batch_size, num_anchors] tensor
                representing the value of the loss function.
        """
        diff = prediction - target
        if self._code_weights is not None:
            diff *= self._code_weights
        abs_diff = paddle.abs(diff)
        abs_diff_lt_1 = (abs_diff <= 1 / (self._sigma**2)).astype(
            abs_diff.dtype)
        loss = abs_diff_lt_1 * 0.5 * paddle.pow(abs_diff * self._sigma, 2) \
               + (abs_diff - 0.5 / (self._sigma ** 2)) * (1. - abs_diff_lt_1)
        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = paddle.sum(loss, 2)
            if weights is not None:
                anchorwise_smooth_l1norm *= weights
        return anchorwise_smooth_l1norm


@manager.LOSSES.add_component
class WeightedSoftmaxClassificationLoss(nn.Layer):
    def __init__(self, logit_scale=1.):
        super(WeightedSoftmaxClassificationLoss, self).__init__()
        self._logit_scale = logit_scale

    def forward(self, prediction, target, weights=None):
        """
        Compute loss function.
        Args:
            prediction: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing the predicted logits for each class
            target: A float tensor of shape [batch_size, num_anchors,
                num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
            loss: a float tensor of shape [batch_size, num_anchors]
                representing the value of the loss function.
        """

        num_classes = prediction.shape[-1]
        prediction /= self._logit_scale

        per_row_cross_ent = F.cross_entropy(
            prediction.reshape((-1, num_classes)),
            target.reshape((-1, 1)),
            reduction="none")

        return per_row_cross_ent.reshape(weights.shape) * weights
