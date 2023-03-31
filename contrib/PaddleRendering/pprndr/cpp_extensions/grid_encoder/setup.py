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

import os

import paddle
from paddle.utils.cpp_extension import BuildExtension, CUDAExtension, setup

major, minor = paddle.device.cuda.get_device_capability()
compute_capability = major * 10 + minor

nvcc_flags = [
    '-O3',
    '-std=c++14',
    # The following definitions must be undefined
    # since TCNN requires half-precision operation.
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
    f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}"
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
else:
    raise NotImplemented("Only support Linux")

c_flags += ['-DPADDLE_WITH_CUDA']

setup(
    name='grid_encoder',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            sources=['src/grid_encoder.cc', 'src/grid_encoder.cu'],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags
            }),
    ],
    cmdclass={'build_ext': BuildExtension})
