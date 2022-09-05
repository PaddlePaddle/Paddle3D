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


class Balancer(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/models/backbones_3d/ffe/ddn_loss/balancer.py#L7
    """

    def __init__(self, fg_weight, bg_weight, downsample_factor=1):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def compute_fg_mask(self, gt_boxes2d, shape, downsample_factor=1):
        """
        Compute foreground mask for images
        Args:
            gt_boxes2d [paddle.Tensor(B, N, 4)]: 2D box labels
            shape [paddle.Size or tuple]: Foreground mask desired shape
            downsample_factor [int]: Downsample factor for image
            device [paddle.device]: Foreground mask desired device
        Returns:
            fg_mask [paddle.Tensor(shape)]: Foreground mask
        """
        fg_mask = paddle.zeros(shape, dtype=paddle.bool)

        # Set box corners
        gt_boxes2d /= downsample_factor
        gt_boxes2d[:, :, :2] = paddle.floor(gt_boxes2d[:, :, :2])
        gt_boxes2d[:, :, 2:] = paddle.ceil(gt_boxes2d[:, :, 2:])
        gt_boxes2d = gt_boxes2d.cast("int64")

        # Set all values within each box to True
        B, N = gt_boxes2d.shape[:2]
        for b in range(B):
            for n in range(N):
                u1, v1, u2, v2 = gt_boxes2d[b, n]
                fg_mask[b, v1:v2, u1:u2] = True

        return fg_mask

    def forward(self, loss, gt_boxes2d):
        """
        Forward pass
        Args:
            loss [paddle.Tensor(B, H, W)]: Pixel-wise loss
            gt_boxes2d [paddle.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
        Returns:
            loss [paddle.Tensor(1)]: Total loss after foreground/background balancing
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        # Compute masks
        fg_mask = self.compute_fg_mask(
            gt_boxes2d=gt_boxes2d,
            shape=loss.shape,
            downsample_factor=self.downsample_factor)
        bg_mask = ~fg_mask
        fg_mask = fg_mask.cast("int64")
        bg_mask = bg_mask.cast("int64")
        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask.cast("bool")].sum() / num_pixels
        bg_loss = loss[bg_mask.cast("bool")].sum() / num_pixels
        # Get total loss
        loss = fg_loss + bg_loss
        tb_dict = {
            "balancer_loss": loss.item(),
            "fg_loss": fg_loss.item(),
            "bg_loss": bg_loss.item()
        }
        return loss, tb_dict
