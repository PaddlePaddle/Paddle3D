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

from paddle3d.models.layers import ConvBNReLU, reset_parameters

from .ddn_loss.ddn_loss import DDNLoss


class FFE(nn.Layer):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/models/backbones_3d/ffe/depth_ffe.py#L9
    """

    def __init__(self, ffe_cfg, disc_cfg):
        """
        Initialize depth classification network
        Args:
            model_cfg [EasyDict]: Depth classification network config
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.disc_cfg = disc_cfg
        self.downsample_factor = ffe_cfg['downsample_factor']

        self.channel_reduce = ConvBNReLU(**ffe_cfg['channel_reduce_cfg'])
        self.ddn_loss = DDNLoss(
            disc_cfg=self.disc_cfg,
            downsample_factor=self.downsample_factor,
            **ffe_cfg['ddn_loss'])
        self.forward_ret_dict = {}
        self.init_weight()

    def forward(self, image_features, depth_logits, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth classification scores
        Args:
            batch_dict:
                images [paddle.Tensor(N, 3, H_in, W_in)]: Input images
        Returns:
            batch_dict:
                frustum_features [paddle.Tensor(N, C, D, H_out, W_out)]: Image depth features
        """
        # Pixel-wise depth classification

        b, c, h, w = paddle.shape(image_features)
        depth_logits = F.interpolate(
            depth_logits, size=[h, w], mode='bilinear', align_corners=False)
        image_features = self.channel_reduce(image_features)
        frustum_features = self.create_frustum_features(
            image_features=image_features, depth_logits=depth_logits)

        batch_dict["frustum_features"] = frustum_features

        if self.training:
            self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]
            self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]
            self.forward_ret_dict["depth_logits"] = depth_logits
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth classification scores
        Args:
            image_features [torch.Tensor(N, C, H, W)]: Image features
            depth_logits [torch.Tensor(N, D, H, W)]: Depth classification logits
        Returns:
            frustum_features [torch.Tensor(N, C, D, H, W)]: Image features
        """
        channel_dim = 1
        depth_dim = 2

        # Resize to match dimensions
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        # Apply softmax along depth axis and remove last depth category (> Max Range)
        depth_probs = F.softmax(depth_logits, axis=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        # Multiply to form image depth feature volume
        frustum_features = depth_probs * image_features
        return frustum_features

    def get_loss(self):
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict

    def init_weight(self):
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                reset_parameters(sublayer)
