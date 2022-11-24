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

from paddle3d.models.layers.layer_libs import _transpose_and_gather_feat


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

        bs_ind = []
        for i in range(pos_pred_pix.shape[0]):
            bs_idx = paddle.full(
                shape=[1, pos_pred_pix.shape[1], 1],
                fill_value=i,
                dtype=ind.dtype)
            bs_ind.append(bs_idx)
        bs_ind = paddle.concat(bs_ind, axis=0)
        m_ind = []
        for i in range(pos_pred_pix.shape[1]):
            m_idx = paddle.full(
                shape=[pos_pred_pix.shape[0], 1, 1],
                fill_value=i,
                dtype=ind.dtype)
            m_ind.append(m_idx)
        m_ind = paddle.concat(m_ind, axis=1)
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
