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
This code is based on https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/models/backbones/second.py
Ths copyright of mmdetection3d is as follows:
Apache-2.0 license [see LICENSE for details].
"""

import math

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Normal, Uniform

from paddle3d.apis import manager
from paddle3d.models.layers.param_init import constant_init, reset_parameters
from paddle3d.models.voxel_encoders.pillar_encoder import build_norm_layer

__all__ = ['SecondBackbone', 'build_conv_layer']


def build_conv_layer(in_channels,
                     out_channels,
                     kernel_size,
                     stride=1,
                     padding=0,
                     dilation=1,
                     groups=1,
                     bias=True,
                     distribution="uniform"):
    """Build convolution layer."""
    if distribution == "uniform":
        bound = 1 / math.sqrt(in_channels * kernel_size**2)
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

    conv_layer = nn.Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        weight_attr=param_attr,
        bias_attr=bias_attr)
    return conv_layer


@manager.BACKBONES.add_component
class SecondBackbone(nn.Layer):
    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 downsample_strides=[2, 2, 2]):
        super(SecondBackbone, self).__init__()
        assert len(downsample_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        self.downsample_strides = downsample_strides

        norm_cfg = dict(type='BatchNorm2D', eps=1e-3, momentum=0.01)
        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=downsample_strides[i],
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels[i]),
                nn.ReLU(),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1,
                        bias=False))
                block.append(build_norm_layer(norm_cfg, out_channels[i]))
                block.append(nn.ReLU())

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.LayerList(blocks)

    def forward(self, x):
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, (nn.Conv1D, nn.Conv2D)):
                reset_parameters(m)
            elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D)):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)

        self.apply(_init_weights)
