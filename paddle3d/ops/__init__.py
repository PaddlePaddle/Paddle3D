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

import importlib
import sys
from types import ModuleType


class CustomOperatorPathFinder:
    def find_module(self, fullname: str, path: str = None):
        if not fullname.startswith('paddle3d.ops'):
            return None

        return CustomOperatorPathLoader()


class CustomOperatorPathLoader:
    def load_module(self, fullname: str):
        modulename = fullname.split('.')[-1]
        sys.modules[fullname] = Paddle3dCustomOperatorModule(
            modulename, fullname)
        return sys.modules[fullname]


class Paddle3dCustomOperatorModule(ModuleType):
    def __init__(self, modulename: str, fullname: str):
        self.fullname = fullname
        self.modulename = modulename
        self.module = None
        super().__init__(modulename)

    def _load_module(self):
        if self.module is None:
            self.module = importlib.import_module(self.modulename)
            sys.modules[self.fullname] = self.module

        return self.module

    def __getattr__(self, attr: str):
        if attr in ['__path__', '__file__']:
            return None

        if attr in ['__loader__', '__package__', '__name__', '__spec__']:
            return super().__getattr__(attr)

        module = self._load_module()
        return getattr(module, attr)


sys.meta_path.insert(0, CustomOperatorPathFinder())
