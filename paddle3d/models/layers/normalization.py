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

import paddle
from paddle import nn


class Scale(nn.Layer):
    """
    This code is based on https://github.com/aim-uofa/AdelaiDet/blob/master/adet/modeling/fcos/fcos.py#L20
    """

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        init_value = paddle.to_tensor([init_value])
        self.scale = paddle.create_parameter(
            shape=init_value.shape,
            dtype='float32',
            default_initializer=nn.initializer.Assign(init_value))

    def forward(self, input):
        return input * self.scale


class Offset(nn.Layer):
    """
    This code is based on https://github.com/TRI-ML/dd3d/blob/main/tridet/layers/normalization.py#L21
    """

    def __init__(self, init_value=0.):
        super(Offset, self).__init__()
        init_value = paddle.to_tensor([init_value])
        self.bias = paddle.create_parameter(
            shape=init_value.shape,
            dtype='float32',
            default_initializer=nn.initializer.Assign(init_value))

    def forward(self, input):
        return input + self.bias


class LayerListDial(nn.LayerList):
    """
    This code is based on https://github.com/aim-uofa/AdelaiDet/blob/master/adet/modeling/fcos/fcos.py#L29
    """

    def __init__(self, layers=None):
        super(LayerListDial, self).__init__(layers)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result
