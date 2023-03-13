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
"""
This code is based on https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/necks/second_fpn.py
Ths copyright of mmdetection3d is as follows:
Apache-2.0 license [see LICENSE for details].
"""

import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Normal, Uniform

from paddle3d.apis import manager
from paddle3d.models.backbones.second_backbone import build_conv_layer
from paddle3d.models.layers.param_init import constant_init, reset_parameters
from paddle3d.models.voxel_encoders.pillar_encoder import build_norm_layer

__all__ = ['SecondFPN']


def build_upsample_layer(in_channels,
                         out_channels,
                         kernel_size,
                         stride=1,
                         padding=0,
                         bias=True,
                         distribution="uniform"):
    """Build upsample layer."""
    if distribution == "uniform":
        bound = 1 / math.sqrt(in_channels)
        param_attr = ParamAttr(initializer=Uniform(-bound, bound))
        bias_attr = False
        if bias:
            bias_attr = ParamAttr(initializer=Uniform(-bound, bound))
    else:
        fan_out = out_channels * kernel_size**2
        std = math.sqrt(2) / math.sqrt(fan_out)
        param_attr = ParamAttr(initializer=Normal(0, std))
        bias_attr = False
        if bias:
            bias_attr = ParamAttr(initializer=Constant(0.))
    deconv_layer = nn.Conv2DTranspose(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        weight_attr=param_attr,
        bias_attr=bias_attr)
    return deconv_layer


class ChannelPool(nn.Layer):
    def forward(self, x):
        return paddle.concat(
            (paddle.max(x, 1).unsqueeze(1), paddle.mean(x, 1).unsqueeze(1)),
            axis=1)


class SpatialGate(nn.Layer):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        norm_cfg = dict(type='BatchNorm2D', eps=1e-5, momentum=0.01)
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            build_conv_layer(
                in_channels=2,
                out_channels=1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
                bias=False), build_norm_layer(norm_cfg, 1, False, False))

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


@manager.NECKS.add_component
class SecondFPN(nn.Layer):
    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 use_conv_for_no_stride=False,
                 use_spatial_attn_before_concat=False):
        super(SecondFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        norm_cfg = dict(type='BatchNorm2D', eps=1e-3, momentum=0.01)
        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False)
            else:
                stride = round(1 / stride)
                upsample_layer = build_conv_layer(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                    bias=False)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel),
                                    nn.ReLU())
            if use_spatial_attn_before_concat:
                deblock.add_sublayer(str(len(deblock)), SpatialGate())
            deblocks.append(deblock)
        self.deblocks = nn.LayerList(deblocks)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, (nn.Conv1D, nn.Conv2D, nn.Conv2DTranspose)):
                reset_parameters(m)
            elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D)):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)

        self.apply(_init_weights)

    def forward(self, x):
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = paddle.concat(ups, axis=1)
        else:
            out = ups[0]
        return out
