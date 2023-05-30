# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import math
import paddle
import paddle.nn as nn
from paddle3d.utils import checkpoint
from paddle3d.utils.logger import logger
from paddle3d.apis import manager

__all__ = ["GUP_DLA", "GUP_DLA34"]


def _make_conv_level(in_channels, out_channels, num_convs, stride=1,
                     dilation=1):
    """
        make conv layers based on its number.
        """
    layers = []
    for i in range(num_convs):
        layers.extend([
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride if i == 0 else 1,
                padding=dilation,
                bias_attr=False,
                dilation=dilation),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        ])

        in_channels = out_channels

    return nn.Sequential(*layers)


class Conv2d(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernal_szie=3,
                 stride=1,
                 bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernal_szie,
            stride=stride,
            padding=kernal_szie // 2,
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Layer):
    """Basic Block
    """

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias_attr=False,
            dilation=dilation)
        self.bn1 = nn.BatchNorm2D(out_channels)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2D(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias_attr=False,
            dilation=dilation)
        self.bn2 = nn.BatchNorm2D(out_channels)

    def forward(self, x, residual=None):
        """forward
        """
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Tree(nn.Layer):
    def __init__(self,
                 level,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False):
        super(Tree, self).__init__()

        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if level == 1:
            self.tree1 = block(
                in_channels, out_channels, stride, dilation=dilation)

            self.tree2 = block(
                out_channels, out_channels, stride=1, dilation=dilation)
        else:
            new_level = level - 1
            self.tree1 = Tree(
                new_level,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)

            self.tree2 = Tree(
                new_level,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
        if level == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.level = level

        self.downsample = None
        if stride > 1:
            self.downsample = nn.MaxPool2D(stride, stride=stride)

        self.project = None
        # If 'self.tree1' is a Tree (not BasicBlock), then the output of project is not used.
        # if in_channels != out_channels and not isinstance(self.tree1, Tree):
        if in_channels != out_channels:  # 和
            self.project = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias_attr=False), nn.BatchNorm2D(out_channels))

    def forward(self, x, residual=None, children=None):
        """forward
        """
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class Root(nn.Layer):
    """Root module
    """

    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias_attr=False,
            padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, *x):
        """forward
        """

        children = x
        x = self.conv(paddle.concat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class GUP_DLABase(nn.Layer):
    """DLA base module
    """

    def __init__(self,
                 levels,
                 channels,
                 block,
                 down_ratio=4,
                 last_level=5,
                 residual_root=False):
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.channels = channels
        self.level_length = len(levels)
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        if block is None:
            block = BasicBlock
        else:
            block = eval(block)

        self.base_layer = nn.Sequential(
            nn.Conv2D(
                3,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias_attr=False), nn.BatchNorm2D(channels[0]), nn.ReLU())

        self.level0 = _make_conv_level(
            in_channels=channels[0],
            out_channels=channels[0],
            num_convs=levels[0])

        self.level1 = _make_conv_level(
            in_channels=channels[0],
            out_channels=channels[1],
            num_convs=levels[1],
            stride=2)

        self.level2 = Tree(
            level=levels[2],
            block=block,
            in_channels=channels[1],
            out_channels=channels[2],
            stride=2,
            level_root=False,
            root_residual=residual_root)

        self.level3 = Tree(
            level=levels[3],
            block=block,
            in_channels=channels[2],
            out_channels=channels[3],
            stride=2,
            level_root=True,
            root_residual=residual_root)

        self.level4 = Tree(
            level=levels[4],
            block=block,
            in_channels=channels[3],
            out_channels=channels[4],
            stride=2,
            level_root=True,
            root_residual=residual_root)

        self.level5 = Tree(
            level=levels[5],
            block=block,
            in_channels=channels[4],
            out_channels=channels[5],
            stride=2,
            level_root=True,
            root_residual=residual_root)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(
                    loc=0., scale=np.sqrt(2. / n),
                    size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    def forward(self, x):
        """forward
        """
        y = []
        x = self.base_layer(x)

        for i in range(self.level_length):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y

    def load_pretrained_model(self, path):
        checkpoint.load_pretrained_model(self, path)


class GUP_DLAUp(nn.Layer):
    """DLA Up module
    """

    def __init__(self, in_channels_list, scales_list=(1, 2, 4, 8, 16)):
        super(GUP_DLAUp, self).__init__()
        scales_list = np.array(scales_list, dtype=int)

        for i in range(len(in_channels_list) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                GUP_IDAUp(
                    in_channels_list=in_channels_list[j:],
                    up_factors_list=scales_list[j:] // scales_list[j],
                    out_channels=in_channels_list[j]))
            scales_list[j + 1:] = scales_list[j]
            in_channels_list[j + 1:] = [
                in_channels_list[j] for _ in in_channels_list[j + 1:]
            ]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            layers[-i - 2:] = ida(layers[-i - 2:])
        return layers[-1]


class GUP_IDAUp(nn.Layer):
    '''
    input: features map of different layers
    output: up-sampled features
    '''

    def __init__(self, in_channels_list, up_factors_list, out_channels):
        super(GUP_IDAUp, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        for i in range(1, len(in_channels_list)):
            in_channels = in_channels_list[i]
            up_factors = int(up_factors_list[i])

            proj = Conv2d(
                in_channels, out_channels, kernal_szie=3, stride=1, bias=False)
            node = Conv2d(
                out_channels * 2,
                out_channels,
                kernal_szie=3,
                stride=1,
                bias=False)
            up = nn.Conv2DTranspose(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=up_factors * 2,
                stride=up_factors,
                padding=up_factors // 2,
                output_padding=0,
                groups=out_channels,
                bias_attr=False)
            # self.fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

        # weight init
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(
                    loc=0., scale=np.sqrt(2. / n),
                    size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    # weight init for up-sample layers [tranposed conv2d]
    def fill_up_weights(self, up):
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, layers):
        assert len(self.in_channels_list) == len(layers), \
            '{} vs {} layers'.format(len(self.in_channels_list), len(layers))

        for i in range(1, len(layers)):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            node = getattr(self, 'node_' + str(i))

            layers[i] = upsample(project(layers[i]))
            layers[i] = node(paddle.concat([layers[i - 1], layers[i]], 1))

        return layers


@manager.BACKBONES.add_component
class GUP_DLA(nn.Layer):
    """DLA base module
    """

    def __init__(self, levels, channels, block, down_ratio=4, pretrained=None):
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]

        self.pretrained = pretrained
        self.first_level = int(np.log2(down_ratio))
        self.base = GUP_DLABase(levels, channels, block)

        # 只加载特征提取网络部分参数
        self.load_pretrained_model()

        scales = [2**i for i in range(len(channels[self.first_level:]))]
        self.dla_up = GUP_DLAUp(
            in_channels_list=channels[self.first_level:], scales_list=scales)

    def forward(self, x):
        """forward
        """
        x = self.base(x)
        feat = self.dla_up(x[self.first_level:])
        return feat

    def load_pretrained_model(self):
        if self.pretrained is not None:
            checkpoint.load_pretrained_model(self, self.pretrained)


@manager.BACKBONES.add_component
def GUP_DLA34(**kwargs):
    model = GUP_DLA(
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        block="BasicBlock",
        **kwargs)

    return model
