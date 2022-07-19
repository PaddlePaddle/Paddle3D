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

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.layers import param_init
from paddle3d.utils import checkpoint

__all__ = ["SACRangeNet21", "SACRangeNet53"]


class SACRangeNet(nn.Layer):
    """
    Backbone of SqueezeSegV3. RangeNet++ architecture with
    Spatially-Adaptive Convolution (SAC).

    For RangeNet++, please refer to:
        Milioto, A., et al. “RangeNet++: Fast and Accurate LiDAR Semantic Segmentation.”
        IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2019.

    For SAC, please refer to:
        Xu, Chenfeng, et al. “SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation.”
        CoRR, vol. abs/2004.01803, 2020, https://arxiv.org/abs/2004.01803.

    Args:
          in_channels (int): The number of channels of input.
          num_layers (int, optional): The depth of SACRangeNet. Defaults to 53.
          encoder_dropout_prob (float, optional): Dropout probability for dropout layers in encoder. Defaults to 0.01.
          decoder_dropout_prob (float, optional): Dropout probability for dropout layers in decoder. Defaults to 0.01.
          bn_momentum (float, optional): Momentum for batch normalization. Defaults to 0.99.
          pretrained (str, optional): Path to pretrained model. Defaults to None.
    """

    # TODO(will-jl944): Currently only SAC-ISK is implemented.

    def __init__(self,
                 in_channels: int,
                 num_layers: int = 53,
                 encoder_dropout_prob: float = .01,
                 decoder_dropout_prob: float = .01,
                 bn_momentum: float = .99,
                 pretrained: str = None):

        supported_layers = {21, 53}
        assert num_layers in supported_layers, "Invalid number of layers ({}) for SACRangeNet backbone, " \
                                               "supported values are {}.".format(num_layers, supported_layers)

        super().__init__()
        self.in_channels = in_channels
        self.pretrained = pretrained

        if num_layers == 21:
            num_stage_blocks = (1, 1, 2, 2, 1)
        elif num_layers == 53:
            num_stage_blocks = (1, 2, 8, 8, 4)

        self.encoder = Encoder(
            in_channels,
            num_stage_blocks,
            encoder_dropout_prob,
            bn_momentum=bn_momentum)
        self.decoder = Decoder(decoder_dropout_prob, bn_momentum=bn_momentum)

        self.init_weight()

    def forward(self, inputs):
        feature, short_cuts = self.encoder(inputs)
        feature_list = self.decoder(feature, short_cuts)

        return feature_list

    def init_weight(self):
        if self.pretrained is not None:
            checkpoint.load_pretrained_model(self, self.pretrained)
        else:
            for layer in self.sublayers():
                if isinstance(layer, (nn.Conv2D, nn.Conv2DTranspose)):
                    param_init.kaiming_uniform_init(
                        layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = param_init._calculate_fan_in_and_fan_out(
                            layer.weight)
                        if fan_in != 0:
                            bound = 1 / math.sqrt(fan_in)
                            param_init.uniform_init(layer.bias, -bound, bound)


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=None,
                 bn_momentum=.9):
        super(ConvBNLayer, self).__init__()
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=bias)

        self._batch_norm = nn.BatchNorm2D(out_channels, momentum=bn_momentum)

    def forward(self, x):
        y = self._conv(x)
        y = self._batch_norm(y)

        return y


class DeconvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=None,
                 bn_momentum=.9):
        super(DeconvBNLayer, self).__init__()
        self._deconv = nn.Conv2DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=bias)

        self._batch_norm = nn.BatchNorm2D(out_channels, momentum=bn_momentum)

    def forward(self, x):
        y = self._deconv(x)
        y = self._batch_norm(y)

        return y


class SACISKBlock(nn.Layer):
    """
    SAC-ISK.
    """

    def __init__(self, num_channels):
        super(SACISKBlock, self).__init__()

        self.attention_layer = ConvBNLayer(
            in_channels=3,
            out_channels=9 * num_channels,
            kernel_size=7,
            padding=3,
            bn_momentum=.9)

        self.position_mlp = nn.Sequential(
            ConvBNLayer(
                in_channels=9 * num_channels,
                out_channels=num_channels,
                kernel_size=1,
                bn_momentum=.9), nn.ReLU(),
            ConvBNLayer(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
                bn_momentum=.9), nn.ReLU())

    def forward(self, xyz, feature):
        N, C, H, W = feature.shape

        new_feature = F.unfold(
            feature, 3, paddings=1).reshape([N, 3 * 3 * C, H, W])
        attention_map = self.attention_layer(xyz)
        attention_map = F.sigmoid(attention_map)
        new_feature = new_feature * attention_map
        new_feature = self.position_mlp(new_feature)
        fused_feature = new_feature + feature

        return xyz, fused_feature


class DownsampleBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, bn_momentum=.9):
        super().__init__()
        self.ds_layer = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=[1, 2],
                padding=1,
                bias=False,
                bn_momentum=bn_momentum), nn.LeakyReLU(.1))

    def forward(self, xyz, feature):
        feature = self.ds_layer(feature)
        xyz = F.interpolate(
            xyz,
            size=[xyz.shape[2], xyz.shape[3] // 2],
            mode="bilinear",
            align_corners=True)

        return xyz, feature


class EncoderStage(nn.Layer):
    def __init__(self,
                 num_blocks,
                 in_channels,
                 out_channels,
                 dropout_prob,
                 downsample=True,
                 bn_momentum=.9):
        super().__init__()

        self.downsample = downsample

        self.layers = nn.LayerList(
            [SACISKBlock(num_channels=in_channels) for _ in range(num_blocks)])

        if downsample:
            self.layers.append(
                DownsampleBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bn_momentum=bn_momentum))

        self.dropout = nn.Dropout2D(dropout_prob)

    def forward(self, xyz, feature):
        for layer in self.layers:
            xyz, feature = layer(xyz, feature)
        feature = self.dropout(feature)

        return xyz, feature


class Encoder(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_stage_blocks=(1, 2, 8, 8, 4),
                 dropout_prob=.01,
                 bn_momentum=.9):
        super(Encoder, self).__init__()

        down_channels = ((32, 64), (64, 128), (128, 256), (256, 256), (256,
                                                                       256))

        self.conv_1 = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                bn_momentum=bn_momentum), nn.LeakyReLU(.1))

        self.encoder_stages = nn.LayerList([
            EncoderStage(
                num_blocks,
                in_ch,
                out_ch,
                dropout_prob=dropout_prob,
                downsample=i < 3,
                bn_momentum=bn_momentum) for i, (num_blocks, (
                    in_ch,
                    out_ch)) in enumerate(zip(num_stage_blocks, down_channels))
        ])

    def forward(self, inputs):
        xyz = inputs[:, 1:4, :, :]
        feature = self.conv_1(inputs)
        short_cuts = []

        for encoder_stage in self.encoder_stages:
            if encoder_stage.downsample:
                short_cuts.append(feature.detach())
            xyz, feature = encoder_stage(xyz, feature)

        return feature, short_cuts


class InvertedResidual(nn.Layer):
    def __init__(self, channels, bn_momentum=.9):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNLayer(
                in_channels=channels[1],
                out_channels=channels[0],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                bn_momentum=bn_momentum), nn.LeakyReLU(.1),
            ConvBNLayer(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                bn_momentum=bn_momentum), nn.LeakyReLU(.1))

    def forward(self, x):
        return self.conv(x) + x


class DecoderStage(nn.Layer):
    def __init__(self, in_channels, out_channels, upsample=True,
                 bn_momentum=.9):
        super().__init__()

        self.upsample = upsample

        self.layers = nn.LayerList()
        if upsample:
            self.layers.append(
                DeconvBNLayer(
                    in_channels,
                    out_channels, [1, 4],
                    stride=[1, 2],
                    padding=[0, 1],
                    bn_momentum=bn_momentum))
        else:
            self.layers.append(
                ConvBNLayer(
                    in_channels,
                    out_channels,
                    3,
                    padding=1,
                    bn_momentum=bn_momentum))

        self.layers.append(nn.LeakyReLU(.1))
        self.layers.append(
            InvertedResidual(
                channels=[in_channels, out_channels], bn_momentum=bn_momentum))

    def forward(self, feature):
        for layer in self.layers:
            feature = layer(feature)

        return feature


class Decoder(nn.Layer):
    def __init__(self, dropout_prob=.01, bn_momentum=.9):
        super().__init__()

        up_channels = ((256, 256), (256, 256), (256, 128), (128, 64), (64, 32))

        self.decoder_stages = nn.LayerList([
            DecoderStage(
                in_ch, out_ch, upsample=i > 1, bn_momentum=bn_momentum)
            for i, (in_ch, out_ch) in enumerate(up_channels)
        ])

        self.dropout = nn.Dropout2D(dropout_prob)

    def forward(self, feature, short_cuts):
        feature_list = []
        for decoder_stage in self.decoder_stages:
            feature = decoder_stage(feature)
            if decoder_stage.upsample:
                feature += short_cuts.pop()
            feature_list.append(self.dropout(feature))

        feature_list[-1] = self.dropout(feature_list[-1])

        return feature_list


@manager.BACKBONES.add_component
def SACRangeNet21(**kwargs) -> paddle.nn.Layer:
    model = SACRangeNet(num_layers=21, **kwargs)

    return model


@manager.BACKBONES.add_component
def SACRangeNet53(**kwargs) -> paddle.nn.Layer:
    model = SACRangeNet(num_layers=53, **kwargs)

    return model
