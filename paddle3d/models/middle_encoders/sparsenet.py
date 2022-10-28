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
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/spconv_backbone.py#L69
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""

import numpy as np
import paddle
from paddle import sparse
from paddle.sparse import nn

from paddle3d.apis import manager
from paddle3d.models.layers import param_init

__all__ = ['SparseNet3D']


def sparse_conv_bn_relu(in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        padding=0,
                        conv_type='subm'):
    if conv_type == 'subm':
        conv = nn.SubmConv3D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias_attr=False)
    elif conv_type == 'spconv':
        conv = nn.Conv3D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=False)
    elif conv_type == 'inverseconv':
        raise NotImplementedError
    else:
        raise NotImplementedError

    m = paddle.nn.Sequential(
        conv,
        nn.BatchNorm(out_channels, epsilon=1e-3, momentum=1 - 0.01),
        nn.ReLU(),
    )

    return m


@manager.MIDDLE_ENCODERS.add_component
class SparseNet3D(paddle.nn.Layer):
    def __init__(self,
                 in_channels=128,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
        super(SparseNet3D, self).__init__()

        self.conv_input = paddle.nn.Sequential(
            nn.SubmConv3D(in_channels, 16, 3, padding=1, bias_attr=False),
            nn.BatchNorm(16, epsilon=1e-3, momentum=1 - 0.01), nn.ReLU())

        self.conv1 = paddle.nn.Sequential(
            sparse_conv_bn_relu(16, 16, 3, padding=1), )

        self.conv2 = paddle.nn.Sequential(
            sparse_conv_bn_relu(
                16, 32, 3, stride=2, padding=1, conv_type='spconv'),
            sparse_conv_bn_relu(32, 32, 3, padding=1),
            sparse_conv_bn_relu(32, 32, 3, padding=1))

        self.conv3 = paddle.nn.Sequential(
            sparse_conv_bn_relu(
                32, 64, 3, stride=2, padding=1, conv_type='spconv'),
            sparse_conv_bn_relu(64, 64, 3, padding=1),
            sparse_conv_bn_relu(64, 64, 3, padding=1))

        self.conv4 = paddle.nn.Sequential(
            sparse_conv_bn_relu(
                64, 64, 3, stride=2, padding=(0, 1, 1), conv_type='spconv'),
            sparse_conv_bn_relu(64, 64, 3, padding=1),
            sparse_conv_bn_relu(64, 64, 3, padding=1),
        )

        last_pad = 0
        self.extra_conv = paddle.nn.Sequential(
            nn.Conv3D(
                64,
                128, (3, 1, 1),
                stride=(2, 1, 1),
                padding=last_pad,
                bias_attr=False),  # [200, 150, 5] -> [200, 150, 2]
            nn.BatchNorm(128, epsilon=1e-3, momentum=1 - 0.01),
            nn.ReLU(),
        )

        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self.sparse_shape = np.array(grid_size[::-1]) + [1, 0, 0]
        self.in_channels = in_channels

        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        self.init_weight()

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, (nn.Conv3D, nn.SubmConv3D)):
                param_init.reset_parameters(layer)
            if isinstance(layer, nn.BatchNorm):
                param_init.constant_init(layer.weight, value=1)
                param_init.constant_init(layer.bias, value=0)

    def forward(self, voxel_features, coors, batch_size):
        shape = [batch_size] + list(self.sparse_shape) + [self.in_channels]
        sp_x = sparse.sparse_coo_tensor(
            coors.transpose((1, 0)),
            voxel_features,
            shape=shape,
            stop_gradient=False)

        x = self.conv_input(sp_x)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        out = self.extra_conv(x_conv4)

        out = out.to_dense()
        out = paddle.transpose(out, perm=[0, 4, 1, 2, 3])
        N, C, D, H, W = out.shape
        out = paddle.reshape(out, shape=[N, C * D, H, W])

        batch_dict = {}
        batch_dict.update({
            'spatial_features': out,
            'spatial_features_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict
