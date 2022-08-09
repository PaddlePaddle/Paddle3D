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
import inspect
import os
import sys
from types import ModuleType

import filelock
from paddle.utils.cpp_extension import load as paddle_jit_load

from paddle3d.env import TMP_HOME
from paddle3d.utils.logger import logger

custom_ops = {
    'voxelize': {
        'sources': ['voxel/voxelize_op.cc', 'voxel/voxelize_op.cu'],
        'version': '0.1.0',
    },
    'iou3d_nms_cuda': {
        'sources': [
            'iou3d_nms/iou3d_cpu.cpp', 'iou3d_nms/iou3d_nms_api.cpp',
            'iou3d_nms/iou3d_nms.cpp', 'iou3d_nms/iou3d_nms_kernel.cu'
        ],
        'version':
        '0.1.0',
    },
    'centerpoint_postprocess': {
        'sources': [
            'centerpoint_postprocess/iou3d_nms_kernel.cu',
            'centerpoint_postprocess/postprocess.cc',
            'centerpoint_postprocess/postprocess.cu'
        ],
        'version':
        '0.1.0',
    },
    'grid_sample_3d': {
        'sources': [
            'grid_sample_3d/grid_sample_3d.cc',
            'grid_sample_3d/grid_sample_3d.cu'
        ],
        'version':
        '0.1.0',
    }
}


class CustomOpNotFoundException(Exception):
    def __init__(self, op_name):
        self.op_name = op_name

    def __str__(self):
        return "Couldn't Found custom op {}".format(self.op_name)


class CustomOperatorPathFinder:
    def find_module(self, fullname: str, path: str = None):
        if not fullname.startswith('paddle3d.ops'):
            return None

        return CustomOperatorPathLoader()


class CustomOperatorPathLoader:
    def load_module(self, fullname: str):
        modulename = fullname.split('.')[-1]

        if modulename not in custom_ops:
            raise CustomOpNotFoundException(modulename)

        if fullname not in sys.modules:
            try:
                sys.modules[fullname] = importlib.import_module(modulename)
            except ImportError:
                sys.modules[fullname] = Paddle3dCustomOperatorModule(
                    modulename, fullname)
        return sys.modules[fullname]


class Paddle3dCustomOperatorModule(ModuleType):
    def __init__(self, modulename: str, fullname: str):
        self.fullname = fullname
        self.modulename = modulename
        self.module = None
        super().__init__(modulename)

    def jit_build(self):
        try:
            lockfile = 'paddle3d.ops.{}'.format(self.modulename)
            lockfile = os.path.join(TMP_HOME, lockfile)
            file = inspect.getabsfile(sys.modules['paddle3d.ops'])
            rootdir = os.path.split(file)[0]

            args = custom_ops[self.modulename].copy()
            sources = args.pop('sources')
            sources = [os.path.join(rootdir, file) for file in sources]

            args.pop('version')
            with filelock.FileLock(lockfile):
                return paddle_jit_load(
                    name=self.modulename, sources=sources, **args)
        except:
            logger.error("{} builded fail!".format(self.modulename))
            raise

    def _load_module(self):
        if self.module is None:
            try:
                self.module = importlib.import_module(self.modulename)
            except ImportError:
                logger.warning("No custom op {} found, try JIT build".format(
                    self.modulename))
                self.module = self.jit_build()
                logger.info("{} builded success!".format(self.modulename))

            # refresh
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
