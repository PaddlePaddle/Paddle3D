# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.vision.models.resnet import BasicBlock, BottleneckBlock
import paddle
from paddle import nn
from paddle3d.apis import manager
from paddle3d.models.layers import param_init, reset_parameters, constant_init

# class BasicBlock(nn.Layer):
#     expansion = 1

#     def __init__(self,
#                  inplanes,
#                  planes,
#                  stride=1,
#                  downsample=None,
#                  groups=1,
#                  base_width=64,
#                  dilation=1,
#                  norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2D

#         if dilation > 1:
#             raise NotImplementedError(
#                 "Dilation > 1 not supported in BasicBlock")

#         self.conv1 = nn.Conv2D(inplanes,
#                                planes,
#                                3,
#                                padding=1,
#                                stride=stride,
#                                bias_attr=False)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#         # self.init_weights()

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

#     # def init_weights(self):
#     #     for m in self.sublayers():
#     #         if isinstance(m, nn.Conv2D):
#     #             reset_parameters(m)
#     #         elif isinstance(m, nn.BatchNorm2D):
#     #             constant_init(m.weight, value=1.0)
#     #             constant_init(m.bias, value=0.0)

# class BottleneckBlock(nn.Layer):

#     expansion = 4

#     def __init__(self,
#                  inplanes,
#                  planes,
#                  stride=1,
#                  downsample=None,
#                  groups=1,
#                  base_width=64,
#                  dilation=1,
#                  norm_layer=None):
#         super(BottleneckBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2D
#         width = int(planes * (base_width / 64.)) * groups

#         self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
#         self.bn1 = norm_layer(width)

#         self.conv2 = nn.Conv2D(width,
#                                width,
#                                3,
#                                padding=dilation,
#                                stride=stride,
#                                groups=groups,
#                                dilation=dilation,
#                                bias_attr=False)
#         self.bn2 = norm_layer(width)

#         self.conv3 = nn.Conv2D(width,
#                                planes * self.expansion,
#                                1,
#                                bias_attr=False)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU()
#         self.downsample = downsample
#         self.stride = stride
#         # self.init_weights()

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

#     # def init_weights(self):
#     #     for m in self.sublayers():
#     #         if isinstance(m, nn.Conv2D):
#     #             reset_parameters(m)
#     #         elif isinstance(m, nn.BatchNorm2D):
#     #             constant_init(m.weight, value=1.0)
#     #             constant_init(m.bias, value=0.0)


@manager.BACKBONES.add_component
class CustomResNet(nn.Layer):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            block_type='Basic',
    ):
        super(CustomResNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BottleneckBlock(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2D(curr_numC, num_channels[i], 3,
                                             stride[i], 1))
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BottleneckBlock(curr_numC, curr_numC // 4)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2D(curr_numC, num_channels[i], 3,
                                             stride[i], 1))
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.layers.apply(param_init.init_weight)

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
