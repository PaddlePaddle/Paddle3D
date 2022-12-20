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

import abc
import codecs
import contextlib
import os
from typing import List, Optional

import yaml
import paddle
import paddle.nn as nn

from paddle3d.slim.quant import QAT


def add_export_args(*args, **kwargs):
    def _wrapper(func):
        if not hasattr(func, 'arg_dict'):
            func.arg_dict = {}

        key = args[0]
        if not key.startswith('--'):
            key = '--{}'.format(key)

        func.arg_dict[key] = kwargs.copy()
        return func

    return _wrapper


class Base3DModel(abc.ABC, nn.Layer):
    def __init__(self):
        super().__init__()
        self.in_export_mode = False
        self._quant = False

    @property
    def input_spec(self) -> paddle.static.InputSpec:
        """Input Tensor specifier when exporting the model."""
        data = {
            _input['name']: paddle.static.InputSpec(**_input)
            for _input in self.inputs
        }

        return [data]

    @abc.abstractproperty
    def inputs(self) -> List[dict]:
        """Model input description. This attribute will be used to construct input_spec."""

    @abc.abstractproperty
    def outputs(self) -> List[dict]:
        """Model output description."""

    def forward(self, samples):
        if self.in_export_mode:
            return self.export_forward(samples)
        elif self.training:
            return self.train_forward(samples)

        return self.test_forward(samples)

    @abc.abstractproperty
    def sensor(self) -> str:
        """The sensor type used in the model sample, usually camera or lidar."""

    def set_export_mode(self, mode: bool = True):
        for sublayer in self.sublayers(include_self=True):
            sublayer.in_export_mode = mode

    @abc.abstractmethod
    def test_forward(self):
        """Test forward function."""

    @abc.abstractmethod
    def train_forward(self):
        """Training forward function."""

    @abc.abstractmethod
    def export_forward(self):
        """Export forward function."""

    @contextlib.contextmanager
    def export_guard(self):
        self.set_export_mode(True)
        yield
        self.set_export_mode(False)

    @property
    def save_name(self):
        return self.__class__.__name__.lower()

    def export(self, save_dir: str, name: Optional[str] = None):
        name = name or self.save_name
        with self.export_guard():
            paddle.jit.to_static(self, input_spec=self.input_spec)
            paddle.jit.save(
                self,
                os.path.join(save_dir, name),
                input_spec=[self.input_spec])

    @property
    def is_quant_model(self) -> bool:
        return self._quant

    def build_slim_model(self, slim_cfg_path: str):
        """ Slim the model and update the cfg params
        """
        self._quant = True

        with codecs.open(slim_cfg_path, 'r', 'utf-8') as f:
            slim_dic = yaml.load(f, Loader=yaml.FullLoader)
            slim_type = slim_dic['slim_type']
            if slim_type == "QAT":
                # create QAT
                quant_config = slim_dic["slim_config"]['quant_config']
                slim = QAT(quant_config=quant_config)
                # slim the model
                slim(self)

            else:
                raise ValueError("slim method `{}` is not supported yet")
