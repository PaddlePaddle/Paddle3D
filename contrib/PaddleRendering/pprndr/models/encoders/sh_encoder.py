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

import paddle
import paddle.nn as nn

from pprndr.apis import manager

try:
    import sh_encoder
except ModuleNotFoundError:
    from pprndr.cpp_extensions import sh_encoder

__all__ = ["SHEncoder"]


@manager.ENCODERS.add_component
class SHEncoder(nn.Layer):
    def __init__(self, input_dim=3, degree=3):
        super(SHEncoder, self).__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = (self.degree + 1)**2

        assert self.input_dim == 3, "SH encoder only support input dim == 3"
        assert 0 <= self.degree < 8, "SH encoder only supports degree in [0, 7]"

    def forward(self, x):
        with paddle.amp.auto_cast(enable=False):
            x = x.reshape([-1, self.input_dim])
            outputs, _ = sh_encoder.sh_encode(
                x, self.degree, (not self.training) or x.stop_gradient)

        return outputs
