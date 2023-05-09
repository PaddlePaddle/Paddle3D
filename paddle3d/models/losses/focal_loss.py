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
import paddle.nn.functional as F
from paddle import nn

from paddle3d.apis import manager
from paddle3d.models.layers.layer_libs import _transpose_and_gather_feat
from paddle3d.models.losses.utils import weight_reduce_loss


class FocalLoss(nn.Layer):
    """Focal loss class
    """

    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        """forward

        Args:
            prediction (paddle.Tensor): model prediction
            target (paddle.Tensor): ground truth

        Returns:
            paddle.Tensor: focal loss
        """
        positive_index = (target == 1).astype("float32")
        negative_index = (target < 1).astype("float32")

        negative_weights = paddle.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = paddle.log(prediction) \
                        * paddle.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = paddle.log(1 - prediction) \
                        * paddle.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive

        return loss


class FastFocalLoss(nn.Layer):
    '''
    This function refers to https://github.com/tianweiy/CenterPoint/blob/cb25e870b271fe8259e91c5d17dcd429d74abc91/det3d/models/losses/centernet_loss.py#L26.
    '''

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, out, target, ind, mask, cat):
        '''
        Arguments:
        out, target: B x C x H x W
        ind, mask: B x M
        cat (category id for peaks): B x M
        '''
        mask = mask.cast('float32')
        gt = paddle.pow(1 - target, 4)
        neg_loss = paddle.log(1 - out) * paddle.pow(out, 2) * gt
        neg_loss = neg_loss.sum()

        pos_pred_pix = _transpose_and_gather_feat(out, ind)  # B x M x C

        index = paddle.arange(start=0, end=pos_pred_pix.shape[0])
        index = index.reshape([pos_pred_pix.shape[0], 1])
        bs_ind = paddle.broadcast_to(
            index, shape=[pos_pred_pix.shape[0], pos_pred_pix.shape[1]])
        bs_ind = bs_ind.reshape(
            [pos_pred_pix.shape[0], pos_pred_pix.shape[1], 1])

        index = paddle.arange(start=0, end=pos_pred_pix.shape[1])
        m_ind = paddle.broadcast_to(
            index, shape=[pos_pred_pix.shape[0], pos_pred_pix.shape[1]])
        m_ind = m_ind.reshape([pos_pred_pix.shape[0], pos_pred_pix.shape[1], 1])

        cat = paddle.concat([bs_ind, m_ind, cat.unsqueeze(2)], axis=-1)
        pos_pred = pos_pred_pix.gather_nd(cat)  # B x M

        num_pos = mask.sum()
        pos_loss = paddle.log(pos_pred) * paddle.pow(1 - pos_pred, 2) * \
                   mask
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


class MultiFocalLoss(nn.Layer):
    """Focal loss class
    """

    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6

    def forward(self, prediction, target):
        """forward

        Args:
            prediction (paddle.Tensor): model prediction
            target (paddle.Tensor): ground truth

        Returns:
            paddle.Tensor: focal loss
        """
        n = prediction.shape[0]

        out_size = [n] + prediction.shape[2:]
        if target.shape[1:] != prediction.shape[2:]:
            raise ValueError(
                f'Expected target size {out_size}, got {target.shape}')

        # compute softmax over the classes axis
        input_soft = F.softmax(prediction, axis=1) + self.eps

        # create the labels one hot tensor
        target_one_hot = F.one_hot(
            target, num_classes=prediction.shape[1]).cast(
                prediction.dtype) + self.eps
        new_shape = [0, len(target_one_hot.shape) - 1
                     ] + [i for i in range(1,
                                           len(target_one_hot.shape) - 1)]

        target_one_hot = target_one_hot.transpose(new_shape)

        # compute the actual focal loss
        weight = paddle.pow(-input_soft + 1.0, self.beta)

        focal = -self.alpha * weight * paddle.log(input_soft)
        loss = paddle.sum(target_one_hot * focal, axis=1)
        # loss = paddle.einsum('bc...,bc...->b...', target_one_hot, focal)
        return loss


class SigmoidFocalClassificationLoss(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/loss_utils.py#L14
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def sigmoid_cross_entropy_with_logits(self, prediction, target):
        """ Implementation for sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = paddle.clip(prediction, min=0) - prediction * target + \
            paddle.log1p(paddle.exp(-paddle.abs(prediction)))
        return loss

    def forward(self, prediction, target, weights):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = F.sigmoid(prediction)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * paddle.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(prediction, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


def sigmoid_focal_loss(inputs, targets, alpha=-1, gamma=2, reduction="none"):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    This code is based on https://github.com/facebookresearch/fvcore/blob/6a5360691be65c76188ed99b57ccbbf5fc19924a/fvcore/nn/focal_loss.py#L7
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.cast('float32')
    targets = targets.cast('float32')
    p = F.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@manager.LOSSES.add_component
class WeightedFocalLoss(nn.Layer):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        This code is modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/focal_loss.py#L160

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(WeightedFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (paddle.Tensor): The prediction.
            target (paddle.Tensor): The learning label of the prediction.
            weight (paddle.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            paddle.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.shape[1]
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]
            loss_cls = self.loss_weight * py_sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """paddle version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    This function is modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/focal_loss.py#L12
    """
    pred_sigmoid = F.sigmoid(pred)
    target = target.astype(pred.dtype)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.shape[0] == loss.shape[0]:
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.reshape([-1, 1])
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.reshape([loss.shape[0], -1])
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss


def gaussian_focal_loss(pred,
                        gaussian_target,
                        weight=None,
                        alpha=2.0,
                        gamma=4.0,
                        reduction='mean',
                        avg_factor=None):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.
    """
    eps = 1e-12
    pos_weights = (gaussian_target == 1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    loss = pos_loss + neg_loss

    return loss


@manager.LOSSES.add_component
class GaussianFocalLoss(nn.Layer):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, alpha=2.0, gamma=4.0, reduction='mean', loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_reg = self.loss_weight * weight_reduce_loss(
            gaussian_focal_loss(
                pred,
                target,
                alpha=self.alpha,
                gamma=self.gamma,
            ),
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg
