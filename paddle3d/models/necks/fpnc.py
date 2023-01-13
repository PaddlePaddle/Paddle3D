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
from paddle3d.models.layers.param_init import (constant_init, reset_parameters,
                                               xavier_uniform_init)

__all__ = ['FPNC', 'ConvModule']


def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='Conv2D')
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    conv_layer = getattr(nn, layer_type)
    layer = conv_layer(*args, **kwargs)
    return layer


def build_norm_layer(cfg, num_features):
    if cfg is None:
        cfg_ = dict(type='BatchNorm2D')
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    norm_layer = getattr(nn, layer_type)
    layer = norm_layer(num_features, **cfg_)
    return layer


def build_activation_layer(cfg):
    if cfg is None:
        cfg_ = dict(type='ReLU')
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    act_layer = getattr(nn, layer_type)
    return act_layer()


class ConvModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.bn = build_norm_layer(norm_cfg, norm_channels)

        # build activation layer
        if self.with_activation:
            self.activate = build_activation_layer(act_cfg)

        self.init_weights()

    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = self.bn(x)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
        return x

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, (nn.Conv1D, nn.Conv2D)):
                reset_parameters(m)
            elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D)):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)

        self.apply(_init_weights)


class FPN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        """
        This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/3f992837ad659f050df38d7b0978372425be16ff/mmdet3d/models/necks/fpn.py
        """
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self, pretrained=None):
        # default init to match torch
        def _init_weights(m):
            if isinstance(m, (nn.Conv1D, nn.Conv2D)):
                reset_parameters(m)
            elif isinstance(m, (nn.BatchNorm1D, nn.BatchNorm2D)):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)

        self.apply(_init_weights)

        # custom init
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                xavier_uniform_init(m.weight)
                if m.bias is not None:
                    constant_init(m.bias, value=0)

    def forward_fpn(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

    def forward(self, inputs):
        return self.forward_fpn(inputs)


@manager.NECKS.add_component
class FPNC(FPN):
    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 final_dim=(900, 1600),
                 downsample=4,
                 use_adp=False,
                 fuse_conv_cfg=None,
                 outC=256,
                 **kwargs):
        """
        This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/3f992837ad659f050df38d7b0978372425be16ff/mmdet3d/models/necks/fpnc.py
        """
        super(FPNC, self).__init__(
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        self.target_size = (final_dim[0] // downsample,
                            final_dim[1] // downsample)
        self.use_adp = use_adp
        if use_adp:
            adp_list = []
            for i in range(self.num_outs):
                if i == 0:
                    resize = nn.AdaptiveAvgPool2D(self.target_size)
                else:
                    resize = nn.Upsample(
                        size=self.target_size,
                        mode='bilinear',
                        align_corners=True)
                adp = nn.Sequential(
                    resize,
                    ConvModule(
                        self.out_channels,
                        self.out_channels,
                        1,
                        padding=0,
                        conv_cfg=fuse_conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                )
                adp_list.append(adp)
            self.adp = nn.LayerList(adp_list)

        self.reduc_conv = ConvModule(
            self.out_channels * self.num_outs,
            outC,
            3,
            padding=1,
            conv_cfg=fuse_conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        outs = self.forward_fpn(x)
        if len(outs) > 1:
            resize_outs = []
            if self.use_adp:
                for i in range(len(outs)):
                    feature = self.adp[i](outs[i])
                    resize_outs.append(feature)
            else:
                target_size = self.target_size
                for i in range(len(outs)):
                    feature = outs[i]
                    if feature.shape[2:] != target_size:
                        feature = F.interpolate(
                            feature,
                            target_size,
                            mode='bilinear',
                            align_corners=True)
                    resize_outs.append(feature)
            out = paddle.concat(resize_outs, axis=1)
            out = self.reduc_conv(out)
        else:
            out = outs[0]
        return [out]
