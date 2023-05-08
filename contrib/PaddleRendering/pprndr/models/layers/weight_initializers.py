#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License")
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import paddle.nn as nn

from pprndr.apis import manager

__all__ = ['GeometricInit']


@manager.WEIGHT_INITIALIZERS.add_component
class GeometricInit(object):
    def __init__(self, bias: float, multi_res: bool):
        self.bias = float(bias)
        self.multi_res = multi_res

    def initialize(self,
                   layers: list,
                   skip_layers: list = None,
                   dims: list = None):
        assert (len(layers) > 1)
        for i, layer in enumerate(layers):
            if i == len(layers) - 1:
                dim = dims[i]  # input_dim of the last layer
                which_layer = "last"

            elif i == 0:
                dim = dims[i + 1]  # output_dim of the first layer
                which_layer = "first"

            elif i in skip_layers:
                dim = [dims[i + 1], (dims[0] - 3)]
                which_layer = "skip"

            else:
                dim = dims[i + 1]
                which_layer = "hidden"

            self._initialize_layer(layer, dim, which_layer)

    def _initialize_layer(self, layer, dim, which_layer):
        assert (which_layer in ["first", "skip", "last", "hidden"])
        if which_layer == "last":
            # Init. weight
            _weight = layer.weight.numpy()
            _weight = np.random.normal(
                np.sqrt(np.pi) / np.sqrt(dim), 0.0001, _weight.shape)
            nn.initializer.Assign(_weight)(layer.weight)
            if layer.bias is not None:
                # Init bias
                _bias = layer.bias.numpy()
                _bias[...] = -self.bias
                nn.initializer.Assign(_bias)(layer.bias)

        elif which_layer == "first" and self.multi_res:
            # Init. weight
            # Note dim should be output dim of this layer
            _weight = layer.weight.numpy()
            _weight = np.random.normal(0.0,
                                       np.sqrt(2) / np.sqrt(dim), _weight.shape)
            _weight[3:, :] = 0.0

            nn.initializer.Assign(_weight)(layer.weight)
            if layer.bias is not None:
                # Init bias
                _bias = layer.bias.numpy()
                _bias[...] = 0
                nn.initializer.Assign(_bias)(layer.bias)

        elif which_layer == "skip" and self.multi_res:
            out_dim = dim[0]
            embed_dim = dim[1]
            # Init weight
            _weight = layer.weight.numpy()
            _weight = np.random.normal(0.0,
                                       np.sqrt(2) / np.sqrt(out_dim),
                                       _weight.shape)
            _weight[-embed_dim:] = 0.0
            nn.initializer.Assign(_weight)(layer.weight)
            if layer.bias is not None:
                # Init bias
                _bias = layer.bias.numpy()
                _bias[...] = 0.0
                nn.initializer.Assign(_bias)(layer.bias)
        else:
            _weight = layer.weight.numpy()
            _weight = np.random.normal(0.0,
                                       np.sqrt(2) / np.sqrt(dim), _weight.shape)
            nn.initializer.Assign(_weight)(layer.weight)
            if layer.bias is not None:
                # Init bias
                _bias = layer.bias.numpy()
                _bias[...] = 0.0
                nn.initializer.Assign(_bias)(layer.bias)
