# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from pprndr.utils.env import COMPUTE_CAPABILITY, TMP_HOME
from pprndr.utils.logger import logger

cpp_extensions = {
    'ffmlp': {
        'sources': ['ffmlp/src/ffmlp.cc', 'ffmlp/src/ffmlp.cu'],
        'version':
        '1.0.0',
        'extra_cxx_cflags': ['-O3', '-std=c++14', '-DPADDLE_WITH_CUDA'],
        'extra_cuda_cflags': [
            '-O3',
            '-std=c++14',
            '--expt-extended-lambda',
            '--expt-relaxed-constexpr',
            '-Xcompiler=-Wno-float-conversion',
            '-Xcompiler=-fno-strict-aliasing',
            f'-gencode=arch=compute_{COMPUTE_CAPABILITY},code=compute_{COMPUTE_CAPABILITY}',
            f'-gencode=arch=compute_{COMPUTE_CAPABILITY},code=sm_{COMPUTE_CAPABILITY}',
            # The following definitions must be undefined
            # since TCNN requires half-precision operation.
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__'
        ],
        'extra_include_paths': [
            'ffmlp/dependencies/cutlass/include',
            'ffmlp/dependencies/cutlass/tools/util/include'
        ]
    },
    'grid_encoder': {
        'sources': [
            'grid_encoder/src/grid_encoder.cc',
            'grid_encoder/src/grid_encoder.cu'
        ],
        'version':
        '1.0.0',
        'extra_cxx_cflags': ['-O3', '-std=c++14', '-DPADDLE_WITH_CUDA'],
        'extra_cuda_cflags': [
            '-O3',
            '-std=c++14',
            '--expt-extended-lambda',
            '--expt-relaxed-constexpr',
            '-Xcompiler=-Wno-float-conversion',
            '-Xcompiler=-fno-strict-aliasing',
            f'-gencode=arch=compute_{COMPUTE_CAPABILITY},code=compute_{COMPUTE_CAPABILITY}',
            f'-gencode=arch=compute_{COMPUTE_CAPABILITY},code=sm_{COMPUTE_CAPABILITY}',
            # The following definitions must be undefined
            # since TCNN requires half-precision operation.
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__'
        ]
    },
    'sh_encoder': {
        'sources':
        ['sh_encoder/src/sh_encoder.cc', 'sh_encoder/src/sh_encoder.cu'],
        'version':
        '1.0.0',
        'extra_cxx_cflags': ['-O3', '-std=c++14', '-DPADDLE_WITH_CUDA'],
        'extra_cuda_cflags': [
            '-O3',
            '-std=c++14',
            f'-gencode=arch=compute_{COMPUTE_CAPABILITY},code=compute_{COMPUTE_CAPABILITY}',
            f'-gencode=arch=compute_{COMPUTE_CAPABILITY},code=sm_{COMPUTE_CAPABILITY}',
            # The following definitions must be undefined
            # since TCNN requires half-precision operation.
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__'
        ]
    },
    'ray_marching_lib': {
        'sources': [
            'ray_marching/src/ray_marching.cc',
            'ray_marching/src/ray_marching.cu',
            'ray_marching/src/rendering.cc',
            'ray_marching/src/rendering.cu',
        ],
        'version':
        '1.0.0',
        'extra_cxx_cflags': ['-O3', '-DPADDLE_WITH_CUDA'],
        'extra_cuda_cflags': ['-O3'],
        'extra_include_paths':
        ['ray_marching/dependencies/cuda-samples/Common']
    },
    'trunc_exp': {
        'sources': ['trunc_exp/src/trunc_exp.cc', 'trunc_exp/src/trunc_exp.cu'],
        'version': '1.0.0',
        'extra_cxx_cflags': ['-DPADDLE_WITH_CUDA'],
    }
}


class CPPExtensionNotFoundException(Exception):
    def __init__(self, op_name):
        self.op_name = op_name

    def __str__(self):
        return "Couldn't Found cpp extension {}".format(self.op_name)


class CPPExtensionPathFinder(object):
    def find_module(self, fullname: str, path: str = None):
        if not fullname.startswith('pprndr.cpp_extensions'):
            return None

        return CPPExtensionPathLoader()


class CPPExtensionPathLoader(object):
    def load_module(self, fullname: str):
        modulename = fullname.split('.')[-1]

        if modulename not in cpp_extensions:
            raise CPPExtensionNotFoundException(modulename)

        if fullname not in sys.modules:
            try:
                sys.modules[fullname] = importlib.import_module(modulename)
            except ImportError:
                sys.modules[fullname] = CPPExtensionModule(modulename, fullname)
        return sys.modules[fullname]


class CPPExtensionModule(ModuleType):
    def __init__(self, modulename: str, fullname: str):
        self.fullname = fullname
        self.modulename = modulename
        self.module = None
        super().__init__(modulename)

    def jit_build(self):
        try:
            lockfile = 'pprndr.cpp_extensions.{}'.format(self.modulename)
            lockfile = os.path.join(TMP_HOME, lockfile)
            file = inspect.getabsfile(sys.modules['pprndr.cpp_extensions'])
            rootdir = os.path.split(file)[0]

            args = cpp_extensions[self.modulename].copy()
            sources = args.pop('sources')
            sources = [os.path.join(rootdir, file) for file in sources]

            include_paths = args.pop('extra_include_paths', None)
            if include_paths is not None:
                include_paths = [
                    os.path.join(rootdir, path) for path in include_paths
                ]

            args.pop('version')
            with filelock.FileLock(lockfile):
                return paddle_jit_load(
                    name=self.modulename,
                    sources=sources,
                    extra_include_paths=include_paths,
                    **args)
        except:
            logger.error("{} building fail!".format(self.modulename))
            raise

    def _load_module(self):
        if self.module is None:
            try:
                self.module = importlib.import_module(self.modulename)
            except ImportError:
                logger.warning(
                    "Cpp extension {} not found, try JIT build".format(
                        self.modulename))
                self.module = self.jit_build()
                logger.info("{} building succeed!".format(self.modulename))

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


sys.meta_path.insert(0, CPPExtensionPathFinder())
