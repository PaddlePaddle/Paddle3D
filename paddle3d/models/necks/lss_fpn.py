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

from paddle import nn
import paddle
from paddle3d.apis import manager
from paddle3d.models.layers import param_init, reset_parameters, constant_init
from paddle3d.models.layers.layer_libs import ConvNormActLayer


@manager.NECKS.add_component
class FPN_LSS(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        channels_factor = 2 if self.extra_upsample else 1
        self.use_input_conv = use_input_conv
        self.input_conv = nn.Sequential(
            nn.Conv2D(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias_attr=False),
            nn.BatchNorm2D(out_channels * channels_factor),
            nn.ReLU(),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2D(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias_attr=False),
            nn.BatchNorm2D(out_channels * channels_factor),
            nn.ReLU(),
            nn.Conv2D(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias_attr=False),
            nn.BatchNorm2D(out_channels * channels_factor),
            nn.ReLU(),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2D(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias_attr=False),
                nn.BatchNorm2D(out_channels),
                nn.ReLU(),
                nn.Conv2D(out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2D(
                    lateral, lateral, kernel_size=1, padding=0,
                    bias_attr=False),
                nn.BatchNorm2D(lateral),
                nn.ReLU(),
            )

        if self.lateral:
            self.lateral_conv.apply(param_init.init_weight)
        if self.input_conv is not None:
            self.input_conv.apply(param_init.init_weight)
        self.conv.apply(param_init.init_weight)

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], \
                 feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x = paddle.concat([x2, x1], axis=1)
        if self.input_conv is not None:
            x = self.input_conv(x)
        x = self.conv(x)
        if self.extra_upsample:
            x = self.up2(x)
        return x
