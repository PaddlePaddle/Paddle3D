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

# code is modified from mmcv: https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/resnet.py
# ------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Normal, Uniform

from paddle3d.apis import manager


class DeformableConvV2(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size**2
        self.mask_channel = kernel_size**2

        offset_bias_attr = ParamAttr(initializer=Constant(0.))

        self.conv_offset = nn.Conv2D(
            in_channels,
            3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(initializer=Constant(0.0)),
            bias_attr=offset_bias_attr)

        self.conv_dcn = paddle.vision.ops.DeformConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            deformable_groups=deformable_groups,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y


def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type_name='Conv2D')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type_name' not in cfg:
            raise KeyError('the cfg dict must contain the key "type_name"')
        cfg_ = copy.deepcopy(cfg)

    layer_type = cfg_.pop('type_name')
    out_channel = args[1]
    kernel_size = args[2]
    fan_out = out_channel * kernel_size**2
    std = math.sqrt(2) / math.sqrt(fan_out)
    param_attr = ParamAttr(initializer=Normal(0, std))
    bias_attr = kwargs.get('bias_attr', True)
    if bias_attr:
        bias_attr = ParamAttr(initializer=Constant(0.))
        kwargs['bias_attr'] = bias_attr

    conv_layer = DeformableConvV2 if layer_type == 'DeformConv2D' else getattr(
        nn, layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


def build_norm_layer(cfg, num_features, postfix='', init_val=1):
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type_name' not in cfg:
        raise KeyError('the cfg dict must contain the key "type_name"')
    cfg_ = copy.deepcopy(cfg)

    layer_type = cfg_.pop('type_name')

    norm_layer = getattr(nn, layer_type)
    abbr = 'bn'
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_['epsilon'] = 1e-5
    weight_attr = ParamAttr(initializer=Constant(value=init_val))
    bias_attr = ParamAttr(initializer=Constant(value=0))
    cfg_['weight_attr'] = weight_attr
    cfg_['bias_attr'] = bias_attr

    layer = norm_layer(num_features, **cfg_)

    if not requires_grad:
        for param in layer.parameters():
            param.trainable = requires_grad

    return name, layer


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type_name='BatchNorm2D'),
                 dcn=None,
                 zero_init_residual=True):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg,
            planes,
            postfix=2,
            init_val=0 if zero_init_residual else 1)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias_attr=False)
        self.add_sublayer(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias_attr=False)
        self.add_sublayer(self.norm2_name, norm2)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type_name='BatchNorm2D'),
                 dcn=None,
                 zero_init_residual=True):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.zero_init_residual = zero_init_residual

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg,
            planes * self.expansion,
            postfix=3,
            init_val=0 if self.zero_init_residual else 1)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            1,
            stride=self.conv1_stride,
            bias_attr=False)
        self.add_sublayer(self.norm1_name, norm1)
        fallback_on_stride = False
        if not self.with_dcn:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias_attr=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias_attr=False)

        self.add_sublayer(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg, planes, planes * self.expansion, 1, bias_attr=False)
        self.add_sublayer(self.norm3_name, norm3)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResLayer(nn.Sequential):
    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type_name='BatchNorm2D'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    1,
                    stride=conv_stride,
                    bias_attr=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


@manager.BACKBONES.add_component
class MMResNet(nn.Layer):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type_name='BatchNorm2D', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 zero_init_residual=True,
                 pretrained=None,
                 lr_factor=1):
        super(MMResNet, self).__init__()
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                zero_init_residual=self.zero_init_residual)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_sublayer(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

        self.init_learning_rate(lr_factor=lr_factor)

    def init_learning_rate(self, lr_factor):
        for _, param in self.named_parameters():
            param.optimize_attr['learning_rate'] = lr_factor

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_sublayer(self.norm1_name, norm1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:

            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.trainable = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.trainable = False

    def forward(self, x):
        """Forward function."""
        if self.training:
            self.train()

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(MMResNet, self).train()
        self._freeze_stages()
        if self.norm_eval:
            for m in self.sublayers():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2D):
                    m.eval()
