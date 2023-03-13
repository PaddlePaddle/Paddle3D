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

from paddle3d.apis import manager
from .utils import weight_reduce_loss


def smooth_l1_loss(input, target, beta, reduction="none"):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:

                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.

    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

    Returns:
        The loss with the reduction option applied.
    This code is based on https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
     """
    beta = paddle.to_tensor(beta).cast(input.dtype)
    if beta < 1e-5:
        loss = paddle.abs(input - target)
    else:
        n = paddle.abs(input - target)
        loss = paddle.where(n < beta, 0.5 * n**2, n - 0.5 * beta)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@manager.LOSSES.add_component
class SmoothL1Loss(nn.Layer):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def smooth_l1_loss(self,
                       pred,
                       target,
                       weight,
                       beta=1.0,
                       reduction='none',
                       avg_factor=None):
        """Smooth L1 loss.

        Args:
            pred (paddle.Tensor): The prediction.
            target (paddle.Tensor): The learning target of the prediction.
            beta (float, optional): The threshold in the piecewise function.
                Defaults to 1.0.

        Returns:
            paddle.Tensor: Calculated loss
        """
        assert beta > 0
        assert pred.shape == target.shape and target.numel() > 0
        diff_ = paddle.abs(pred - target)
        loss = paddle.where(diff_ < beta, 0.5 * diff_ * diff_ / beta,
                            diff_ - 0.5 * beta)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (paddle.Tensor): The prediction.
            target (paddle.Tensor): The learning target of the prediction.
            weight (paddle.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)

        loss_bbox = self.loss_weight * self.smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
