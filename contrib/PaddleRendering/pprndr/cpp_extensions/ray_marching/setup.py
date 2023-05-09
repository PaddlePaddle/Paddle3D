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

from paddle.utils.cpp_extension import BuildExtension, CUDAExtension, setup

nvcc_flags = ['-O3', '-std=c++14']

if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']
else:
    raise NotImplemented("Only support Linux")

c_flags += ['-DPADDLE_WITH_CUDA']

setup(
    name='ray_marching_lib',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            sources=[
                'src/ray_marching.cc', 'src/ray_marching.cu',
                'src/rendering.cc', 'src/rendering.cu'
            ],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags
            },
            include_dirs=['dependencies/cuda-samples/Common']),
    ],
    cmdclass={'build_ext': BuildExtension})
