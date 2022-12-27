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

from collections import OrderedDict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.layers import group_norm, FrozenBatchNorm2d, param_init
from paddle3d.utils import checkpoint

__all__ = ["VoVNet", "VoVNet99_eSE"]

norm_func = None


def dw_conv3x3(in_channels,
               out_channels,
               module_name,
               postfix,
               stride=1,
               kernel_size=3,
               padding=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        ('{}_{}/dw_conv3x3'.format(module_name, postfix),
         nn.Conv2D(
             in_channels,
             out_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             groups=out_channels,
             bias_attr=False)),
        ('{}_{}/pw_conv1x1'.format(module_name, postfix),
         nn.Conv2D(
             in_channels,
             out_channels,
             kernel_size=1,
             stride=1,
             padding=0,
             groups=1,
             bias_attr=False)),
        ('{}_{}/pw_norm'.format(module_name, postfix), norm_func(out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU()))


def conv3x3(in_channels,
            out_channels,
            module_name,
            postfix,
            stride=1,
            groups=1,
            kernel_size=3,
            padding=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        (f"{module_name}_{postfix}/conv",
         nn.Conv2D(
             in_channels,
             out_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             groups=groups,
             bias_attr=False,
         )), (f"{module_name}_{postfix}/norm", norm_func(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU()))


def conv1x1(in_channels,
            out_channels,
            module_name,
            postfix,
            stride=1,
            groups=1,
            kernel_size=1,
            padding=0):
    """1x1 convolution with padding"""
    return nn.Sequential(
        (f"{module_name}_{postfix}/conv",
         nn.Conv2D(
             in_channels,
             out_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             groups=groups,
             bias_attr=False,
         )), (f"{module_name}_{postfix}/norm", norm_func(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU()))


class Hsigmoid(nn.Layer):
    def __init__(self):
        super(Hsigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3.0) / 6.0


class eSEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Conv2D(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Layer):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 SE=False,
                 identity=False,
                 depthwise=False):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.LayerList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = conv1x1(
                in_channel, stage_ch, "{}_reduction".format(module_name), "0")
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(
                    dw_conv3x3(stage_ch, stage_ch, module_name, i))
            else:
                self.layers.append(
                    conv3x3(in_channel, stage_ch, module_name, i))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = conv1x1(in_channel, concat_ch, module_name, "concat")

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = paddle.concat(output, axis=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num,
                 SE=False,
                 depthwise=False):

        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_sublayer(
                "Pooling", nn.MaxPool2D(
                    kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        self.add_sublayer(
            module_name,
            _OSA_module(
                in_ch,
                stage_ch,
                concat_ch,
                layer_per_block,
                module_name,
                SE,
                depthwise=depthwise))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_sublayer(
                module_name,
                _OSA_module(
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    SE,
                    identity=True,
                    depthwise=depthwise),
            )


@manager.BACKBONES.add_component
class VoVNet(nn.Layer):
    def __init__(self,
                 stem_ch,
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 depthwise,
                 SE,
                 norm_type,
                 input_ch,
                 out_features=None):
        """
        Args:
            input_ch(int) : the number of input channel
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "stage2" ...
        """
        super(VoVNet, self).__init__()

        global norm_func
        if norm_type == "bn" or norm_type is None:
            norm_func = nn.BatchNorm2D
        elif norm_type == "gn":
            norm_func = group_norm
        elif norm_type == "frozen_bn":
            norm_func = FrozenBatchNorm2d
        else:
            raise NotImplementedError()

        self._out_features = out_features

        # Stem module
        conv_type = dw_conv3x3 if depthwise else conv3x3
        self.stem = nn.Sequential(('stem1',
                                   conv3x3(input_ch, stem_ch[0], "stem", "1",
                                           2)))
        self.stem.add_sublayer(
            'stem2', conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1))
        self.stem.add_sublayer(
            'stem3', conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2))
        current_stirde = 4
        self._out_feature_strides = {
            "stem": current_stirde,
            "stage2": current_stirde
        }
        self._out_feature_channels = {"stem": stem_ch[2]}

        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        # OSA stages
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "stage%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_sublayer(
                name,
                _OSA_stage(
                    in_ch_list[i],
                    config_stage_ch[i],
                    config_concat_ch[i],
                    block_per_stage[i],
                    layer_per_block,
                    i + 2,
                    SE,
                    depthwise,
                ),
            )

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stirde = int(
                    current_stirde * 2)

        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                param_init.kaiming_normal_init(m.weight)

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs.append(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs.append(x)

        return outputs


@manager.BACKBONES.add_component
def VoVNet99_eSE(**kwargs):

    model = VoVNet(
        stem_ch=[64, 64, 128],
        config_stage_ch=[128, 160, 192, 224],
        config_concat_ch=[256, 512, 768, 1024],
        layer_per_block=5,
        block_per_stage=[1, 3, 9, 3],
        SE=True,
        depthwise=False,
        **kwargs)

    return model
