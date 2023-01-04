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
This code is based on https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/backbones/scn.py
Ths copyright of tianweiy/CenterPoint is as follows:
MIT License [see LICENSE for details].
"""

import numpy as np
import paddle
from paddle import sparse
from paddle.sparse import nn

from paddle3d.apis import manager
from paddle3d.models.layers import param_init

__all__ = ['SparseResNet3D']


def conv3x3(in_out_channels,
            out_out_channels,
            stride=1,
            indice_key=None,
            bias_attr=True):
    """3x3 convolution with padding"""
    return nn.SubmConv3D(
        in_out_channels,
        out_out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=bias_attr,
        key=indice_key)


def conv1x1(in_out_channels,
            out_out_channels,
            stride=1,
            indice_key=None,
            bias_attr=True):
    """1x1 convolution"""
    return nn.SubmConv3D(
        in_out_channels,
        out_out_channels,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias_attr=bias_attr,
        key=indice_key)


class SparseBasicBlock(paddle.nn.Layer):
    expansion = 1

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            downsample=None,
            indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        bias_attr = True

        self.conv1 = conv3x3(
            in_channels,
            out_channels,
            stride,
            indice_key=indice_key,
            bias_attr=bias_attr)
        self.bn1 = nn.BatchNorm(out_channels, epsilon=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(
            out_channels,
            out_channels,
            indice_key=indice_key,
            bias_attr=bias_attr)
        self.bn2 = nn.BatchNorm(out_channels, epsilon=1e-3, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = sparse.add(out, identity)
        out = self.relu(out)

        return out


@manager.MIDDLE_ENCODERS.add_component
class SparseResNet3D(paddle.nn.Layer):
    def __init__(self,
                 in_channels=128,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1)):
        super(SparseResNet3D, self).__init__()

        self.zero_init_residual = False

        # input: # [1600, 1200, 41]
        self.conv_input = paddle.nn.Sequential(
            nn.SubmConv3D(in_channels, 16, 3, bias_attr=False, key='res0'),
            nn.BatchNorm(16, epsilon=1e-3, momentum=0.01), nn.ReLU())

        self.conv1 = paddle.nn.Sequential(
            SparseBasicBlock(16, 16, indice_key='res0'),
            SparseBasicBlock(16, 16, indice_key='res0'),
        )

        self.conv2 = paddle.nn.Sequential(
            nn.Conv3D(16, 32, 3, 2, padding=1,
                      bias_attr=False),  # [1600, 1200, 41] -> [800, 600, 21]
            nn.BatchNorm(32, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            SparseBasicBlock(32, 32, indice_key='res1'),
            SparseBasicBlock(32, 32, indice_key='res1'),
        )

        self.conv3 = paddle.nn.Sequential(
            nn.Conv3D(32, 64, 3, 2, padding=1,
                      bias_attr=False),  # [800, 600, 21] -> [400, 300, 11]
            nn.BatchNorm(64, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            SparseBasicBlock(64, 64, indice_key='res2'),
            SparseBasicBlock(64, 64, indice_key='res2'),
        )

        self.conv4 = paddle.nn.Sequential(
            nn.Conv3D(64, 128, 3, 2, padding=[0, 1, 1],
                      bias_attr=False),  # [400, 300, 11] -> [200, 150, 5]
            nn.BatchNorm(128, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
            SparseBasicBlock(128, 128, indice_key='res3'),
            SparseBasicBlock(128, 128, indice_key='res3'),
        )

        self.extra_conv = paddle.nn.Sequential(
            nn.Conv3D(128, 128, (3, 1, 1), (2, 1, 1),
                      bias_attr=False),  # [200, 150, 5] -> [200, 150, 2]
            nn.BatchNorm(128, epsilon=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self.sparse_shape = np.array(grid_size[::-1]) + [1, 0, 0]
        self.in_channels = in_channels
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
        return out
