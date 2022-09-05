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

from paddle3d.models.losses import MultiFocalLoss
from paddle3d.utils.depth import bin_depths

from .balancer import Balancer


class DDNLoss(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py#L9
    """

    def __init__(self, weight, alpha, beta, disc_cfg, fg_weight, bg_weight,
                 downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            beta [float]: Beta value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.disc_cfg = disc_cfg
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.beta = beta
        self.loss_func = MultiFocalLoss(alpha=self.alpha, beta=self.beta)
        self.weight = weight

    def forward(self, depth_logits, depth_maps, gt_boxes2d):
        """
        Gets DDN loss
        Args:
            depth_logits: paddle.Tensor(B, D+1, H, W)]: Predicted depth logits
            depth_maps: paddle.Tensor(B, H, W)]: Depth map [m]
            gt_boxes2d [paddle.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
        Returns:
            loss [paddle.Tensor(1)]: Depth classification network loss
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        tb_dict = {}

        # Bin depth map to create target
        depth_target = bin_depths(depth_maps, **self.disc_cfg, target=True)

        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)

        # Compute foreground/background balancing
        loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)
        # Final loss
        loss *= self.weight
        tb_dict.update({"ddn_loss": loss.item()})

        return loss, tb_dict
