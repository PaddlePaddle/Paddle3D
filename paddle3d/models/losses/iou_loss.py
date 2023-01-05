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
from paddle import nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.losses.utils import weight_reduce_loss
from paddle3d.utils.box import bbox_overlaps


class IOULoss(nn.Layer):
    """
    Intersetion Over Union (IoU) loss
    This code is based on https://github.com/aim-uofa/AdelaiDet/blob/master/adet/layers/iou_loss.py
    """

    def __init__(self, loc_loss_type='iou'):
        """
        Args:
            loc_loss_type: str, supports three IoU computations: 'iou', 'linear_iou', 'giou'.
        """
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = paddle.minimum(pred_left, target_left) + \
                      paddle.minimum(pred_right, target_right)
        h_intersect = paddle.minimum(pred_bottom, target_bottom) + \
                      paddle.minimum(pred_top, target_top)

        g_w_intersect = paddle.maximum(pred_left, target_left) + \
                        paddle.maximum(pred_right, target_right)
        g_h_intersect = paddle.maximum(pred_bottom, target_bottom) + \
                        paddle.maximum(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -paddle.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()


def giou_loss(pred, target, weight, eps=1e-7, reduction='mean',
              avg_factor=None):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    This function is modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py#L102

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@manager.LOSSES.add_component
class GIoULoss(nn.Layer):
    '''
    This class is modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py#L358
    '''

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not paddle.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze([1])
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
