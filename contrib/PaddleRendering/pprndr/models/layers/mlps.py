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

import math
from typing import List, Tuple, Union, Callable, Any

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pprndr.apis import manager

try:
    import ffmlp
except ModuleNotFoundError:
    from pprndr.cpp_extensions import ffmlp

__all__ = ['FFMLP', 'MLP']

TCNN_ACTIVATION = {
    "relu": 0,
    "exponential": 1,
    "sine": 2,
    "sigmoid": 3,
    "squareplus": 4,
    "softplus": 5,
    "none": 6
}

PADDLE_ACTIVATION = {
    "relu":
    F.relu,
    "exponential":
    paddle.exp,
    "sine":
    paddle.sin,
    "sigmoid":
    F.sigmoid,
    "squareplus":
    lambda x: .5 * (10. * x + paddle.sqrt(paddle.square(10. * x) + 4.)) / 10.,
    "softplus":
    F.softplus,
    "none":
    None
}


@manager.LAYERS.add_component
class FFMLP(nn.Layer):
    supported_hidden_dim = (16, 32, 64, 128, 256)

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 activation: str = "relu",
                 output_activation: str = "none"):
        super(FFMLP, self).__init__()

        assert hidden_dim in self.supported_hidden_dim, \
            f"FFMLP only supports hidden_dim in {self.supported_hidden_dim}, but got {hidden_dim}"
        assert input_dim > 0 and input_dim % 16 == 0, \
            f"input_dim must be positive and a multiple of 16, but got {input_dim}"
        assert 0 < output_dim <= 16, f"output_dim must be in (0, 16], but got {output_dim}"
        assert num_layers >= 2, f"num_layers must be at least 2, but got {num_layers}"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = TCNN_ACTIVATION[activation.lower()]
        self.output_activation = PADDLE_ACTIVATION[output_activation.lower()]

        # pad output
        self.padded_output_dim = int(math.ceil(output_dim / 16)) * 16

        num_parameters = hidden_dim * (
            input_dim + hidden_dim * (num_layers - 2) + self.padded_output_dim)
        std = math.sqrt(3 / hidden_dim)
        self.weights = self.create_parameter(
            [num_parameters],
            dtype="float16",
            default_initializer=nn.initializer.Uniform(-std, std))

        ffmlp.allocate_splitk(self.num_layers)

    def forward(self, x):
        with paddle.amp.auto_cast(enable=False):
            x = x.astype("float16")

            input_shape = x.shape
            C = input_shape[-1]
            x = x.reshape([-1, C])
            B = x.shape[0]

            channel_pad_size = self.input_dim - C
            batch_pad_size = (B + 128 - 1) // 128 * 128 - B

            if channel_pad_size > 0 or batch_pad_size > 0:
                x = F.pad(x, [0, batch_pad_size, 0, channel_pad_size])

            if self.training:
                outputs, _ = ffmlp.ffmlp_op(
                    x, self.weights, self.padded_output_dim, self.hidden_dim,
                    self.num_layers - 1, self.activation,
                    TCNN_ACTIVATION["none"])
            else:
                outputs = ffmlp.ffmlp_infer_op(
                    x, self.weights, self.padded_output_dim, self.hidden_dim,
                    self.num_layers - 1, self.activation,
                    TCNN_ACTIVATION["none"])

            # unpad output
            if batch_pad_size > 0:
                outputs = outputs[:B]
            if self.padded_output_dim != self.output_dim:
                outputs = outputs[:, :self.output_dim]

            outputs = outputs.reshape(input_shape[:-1] + [self.output_dim])
            outputs = outputs.astype("float32")

            if self.output_activation is not None:
                outputs = self.output_activation(outputs)

        return outputs


@manager.LAYERS.add_component
class MLP(nn.Layer):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_layers: int,
                 hidden_dim: Union[int, List[int], Tuple[int]] = None,
                 skip_layers: Union[int, List[int], Tuple[int]] = None,
                 skip_connection_way: str = "concat_out",
                 skip_connection_scale: float = None,
                 with_bias: bool = True,
                 activation: Union[str, nn.Layer] = "relu",
                 output_activation: Union[str, nn.Layer] = "none",
                 weight_init: Any = None,
                 weight_normalization: bool = False):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.with_bias = with_bias
        self.skip_scale = skip_connection_scale

        if num_layers > 1:
            assert hidden_dim is not None, "hidden_dim must be specified when num_layers > 1"
            if isinstance(hidden_dim, int):
                hidden_dim = [hidden_dim] * (num_layers - 1)
            else:
                assert len(
                    hidden_dim
                ) == num_layers - 1, "hidden_dim must have the same length as num_layers - 1"

            if skip_layers is not None:
                if isinstance(skip_layers, int):
                    skip_layers = {skip_layers}
                else:
                    skip_layers = set(skip_layers)

        else:
            assert hidden_dim is None, "hidden_dim must be None when num_layers == 1"
            assert skip_layers is None, "skip_layers must be None when num_layers == 1"

        self.hidden_dim = hidden_dim
        self.skip_layers = skip_layers if skip_layers is not None else set()
        if len(self.skip_layers) > 0 and skip_connection_way == "concat_in":
            for skip_layer in self.skip_layers:
                self.hidden_dim[skip_layer - 1] -= self.input_dim

        self.activation = self._populate_activation(activation)
        self.output_activation = self._populate_activation(output_activation)
        self.layers = self._populate_layers()

        # Weight init_fn.
        if weight_init is not None:
            weight_init.initialize(self.layers, self.skip_layers,
                                   [self.input_dim] + self.hidden_dim)

        # Weight normalization
        if weight_normalization:
            for layer in self.layers:
                nn.utils.weight_norm(layer, dim=1)

    def _populate_activation(self,
                             activation: Union[str, nn.Layer]) -> Callable:
        if isinstance(activation, str):
            activation = PADDLE_ACTIVATION[activation.lower()]
        elif not isinstance(activation, nn.Layer):
            raise NotImplementedError(
                "activation should be either str or nn.Layer.")

        return activation

    def _populate_layers(self) -> nn.LayerList:
        layers = nn.LayerList()
        if self.num_layers == 1:
            layers.append(
                nn.Linear(
                    self.input_dim, self.output_dim, bias_attr=self.with_bias))
        else:
            # First layer
            layers.append(
                nn.Linear(
                    self.input_dim,
                    self.hidden_dim[0],
                    bias_attr=self.with_bias))

            # hidden layers
            for i in range(1, self.num_layers - 1):
                if i in self.skip_layers:
                    layers.append(
                        nn.Linear(
                            self.hidden_dim[i - 1] + self.input_dim,
                            self.hidden_dim[i],
                            bias_attr=self.with_bias))
                else:
                    layers.append(
                        nn.Linear(
                            self.hidden_dim[i - 1],
                            self.hidden_dim[i],
                            bias_attr=self.with_bias))
            # Last layer
            layers.append(
                nn.Linear(
                    self.hidden_dim[-1],
                    self.output_dim,
                    bias_attr=self.with_bias))

        return layers

    def forward(self, x):
        x = x.astype("float32")
        residual = x
        for i, layer in enumerate(self.layers[:-1]):
            if i in self.skip_layers:
                x = paddle.concat([x, residual], axis=-1)
                if self.skip_scale is not None:
                    x = x / self.skip_scale
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)

        x = self.layers[-1](x).astype("float32")

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x
