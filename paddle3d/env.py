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

import glob
import importlib
import os
import platform
import subprocess
import sys
from typing import List, Optional

import cv2
import paddle
import paddleseg

import paddle3d


def init_distributed():
    """
    """
    if not is_distributed_inited():
        paddle.distributed.fleet.init(is_collective=True)


def is_distributed_inited():
    """
    """
    return paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
    )


def get_package_version(package: str) -> str:
    """
    """
    try:
        module = importlib.import_module(package)
        version = module.__version__
    except:
        version = 'Not Installed'

    return version


def get_envrionment_flags(FLAG: str) -> str:
    return os.environ.get(FLAG, 'Not set.')


def get_gcc_info() -> str:
    """
    """
    try:
        gcc = subprocess.check_output(['gcc', '--version']).decode()
        gcc = gcc.strip().split('\n')[0]
    except:
        gcc = 'Not Found.'

    return gcc


def get_nvcc_info(cuda_home):
    if cuda_home is not None and os.path.isdir(cuda_home):
        try:
            nvcc = os.path.join(cuda_home, 'bin/nvcc')
            nvcc = subprocess.check_output(
                "{} -V".format(nvcc), shell=True).decode()
            nvcc = nvcc.strip().split('\n')[-1]
        except subprocess.SubprocessError:
            nvcc = "Not Available"
    else:
        nvcc = "Not Available"
    return nvcc


def get_cuda_device_info(devices: Optional[List[int]] = None) -> List[str]:
    if devices is None:
        try:
            devices = get_envrionment_flags('CUDA_VISIBLE_DEVICES')
            devices = [int(device) for device in devices.split(',')]
        except:
            devices = []

    try:
        cmds = ['nvidia-smi', '-L']
        gpu_info = subprocess.check_output(cmds)
        gpu_info = gpu_info.decode().strip().split('\n')
        gpu_info = [' '.join(gpu_info[i].split(' ')[:4]) for i in devices]
    except:
        gpu_info = ['Not Found.']

    return gpu_info


def get_env_info():
    msgs = []
    msgs.append('------------Environment Information-------------')

    # add platform info
    msgs.append('platform:')
    msgs.append('    {}'.format(platform.platform()))
    msgs.append('    {}'.format(get_gcc_info()))
    msgs.append('    Python - {}'.format(sys.version.replace('\n', ' ')))

    # add Science Toolkits info
    st_pakcages = {
        'cv2': get_package_version('cv2'),
        'numpy': get_package_version('numpy'),
        'numba': get_package_version('numba'),
        'pandas': get_package_version('pandas'),
        'pillow': get_package_version('PIL'),
        'skimage': get_package_version('skimage')
    }

    msgs.append('\nScience Toolkits:')
    for package, version in st_pakcages.items():
        msgs.append('    {} - {}'.format(package, version))

    if paddle.is_compiled_with_cuda():
        _paddle = 'paddle(gpu)'
    else:
        _paddle = 'paddle'
    paddle_packages = {
        _paddle: paddle.__version__,
        'paddle3d': paddle3d.__version__,
        'paddleseg': paddleseg.__version__
    }

    paddle_flags = [
        'FLAGS_cudnn_deterministic', 'FLAGS_cudnn_exhaustive_search'
    ]

    # add Paddle info
    msgs.append('\nPaddlePaddle:')
    for package, version in paddle_packages.items():
        msgs.append('    {} - {}'.format(package, version))

    for flag in paddle_flags:
        msgs.append('    {} - {}'.format(flag, get_envrionment_flags(flag)))

    # add CUDA info
    msgs.append('\nCUDA:')
    msgs.append('    cudnn - {}'.format(paddle.get_cudnn_version()))
    msgs.append('    nvcc - {}'.format(get_nvcc_info(get_cuda_home())))
    # TODO: Add nccl version

    # add GPU info
    msgs.append('\nGPUs:')
    for device in get_cuda_device_info():
        msgs.append('    {}'.format(device))

    msgs.append('------------------------------------------------')
    return '\n'.join(msgs)


def get_cuda_home():
    '''Finds the CUDA install path. It refers to the implementation of
    pytorch <https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py>.
    '''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            nvcc = subprocess.check_output([which, 'nvcc'],
                                           stderr=subprocess.STDOUT)
            cuda_home = os.path.dirname(
                os.path.dirname(nvcc.decode().rstrip('\r\n')))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    return cuda_home


def get_user_home() -> str:
    return os.path.expanduser('~')


def get_paddle3d_home() -> str:
    return os.path.join(get_user_home(), '.paddle3d')


def get_sub_home(directory: str) -> str:
    home = os.path.join(get_paddle3d_home(), directory)
    os.makedirs(home, exist_ok=True)
    return home


USER_HOME = get_user_home()
PADDLE3D_HOME = get_paddle3d_home()
PRETRAINED_HOME = get_sub_home('pretrained')
TMP_HOME = get_sub_home('tmp')

IS_WINDOWS = sys.platform == 'win32'
nranks = paddle.distributed.ParallelEnv().nranks
local_rank = paddle.distributed.ParallelEnv().local_rank

# supress Numba warnings
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0"
