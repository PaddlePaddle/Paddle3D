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

from paddle3d.models.layers import ConvBNReLU, reset_parameters


class BEV(nn.Layer):
    def __init__(self, layer_nums, layer_strides, num_filters, upsample_strides,
                 num_upsample_filters, input_channels):
        super().__init__()

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.Sequential()
        self.deblocks = nn.Sequential()
        for idx in range(num_levels):
            cur_layers = [
                nn.Pad2D(1),
                nn.Conv2D(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias_attr=False),
                nn.BatchNorm2D(num_filters[idx], epsilon=1e-3, momentum=0.99),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2D(
                        num_filters[idx],
                        num_filters[idx],
                        kernel_size=3,
                        padding=1,
                        bias_attr=False),
                    nn.BatchNorm2D(
                        num_filters[idx], epsilon=1e-3, momentum=0.99),
                    nn.ReLU()
                ])
            self.blocks.add_sublayer("level_" + str(idx),
                                     nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.add_sublayer(
                        "level_" + str(idx),
                        nn.Sequential(
                            nn.Conv2DTranspose(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                upsample_strides[idx],
                                stride=upsample_strides[idx],
                                bias_attr=False),
                            nn.BatchNorm2D(
                                num_upsample_filters[idx],
                                epsilon=1e-3,
                                momentum=0.99), nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.add_sublayer(
                        "level_" + str(idx),
                        nn.Sequential(
                            nn.Conv2D(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                stride,
                                stride=stride,
                                bias_attr=False),
                            nn.BatchNorm2D(
                                num_upsample_filters[idx],
                                epsilon=1e-3,
                                momentum=0.99), nn.ReLU()))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.add_sublayer(
                "upsample",
                nn.Sequential(
                    nn.Conv2DTranspose(
                        c_in,
                        c_in,
                        upsample_strides[-1],
                        stride=upsample_strides[-1],
                        bias_attr=False),
                    nn.BatchNorm2D(c_in, epsilon=1e-3, momentum=0.99),
                    nn.ReLU(),
                ))

        self.num_bev_features = c_in
        self.init_weight()

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks["level_" + str(i)](x)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks["level_" + str(i)](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = paddle.concat(ups, axis=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks["upsample"](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

    def init_weight(self):
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                reset_parameters(sublayer)
