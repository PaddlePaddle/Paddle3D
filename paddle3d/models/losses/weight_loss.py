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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager

from paddle3d.models.common import boxes_to_corners_3d


class WeightedCrossEntropyLoss(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/loss_utils.py#L187
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, prediction, target, weights):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        target = target.argmax(axis=-1)
        loss = F.cross_entropy(prediction, target, reduction='none') * weights
        return loss


@manager.LOSSES.add_component
class WeightedSmoothL1Loss(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/loss_utils.py#L80

                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta=1.0 / 9.0, code_weights=None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.code_weights = paddle.to_tensor(code_weights)

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = paddle.abs(diff)
        else:
            n_diff = paddle.abs(diff)
            loss = paddle.where(n_diff < beta, 0.5 * n_diff**2 / beta,
                                n_diff - 0.5 * beta)

        return loss

    def forward(self, input, target, weights=None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = paddle.where(paddle.isnan(target), input,
                              target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.reshape([1, 1, -1])

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[
                1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)
        return loss


def get_corner_loss_lidar(pred_bbox3d, gt_bbox3d):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = paddle.minimum(
        paddle.linalg.norm(pred_box_corners - gt_box_corners, axis=2),
        paddle.linalg.norm(pred_box_corners - gt_box_corners_flip, axis=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(axis=1)


@manager.LOSSES.add_component
class WeightedL1Loss(nn.Layer):
    """
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, input, target, weight=None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """

        loss = self.loss(input, target)
        if weight is not None:
            loss *= weight

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss * self.loss_weight
