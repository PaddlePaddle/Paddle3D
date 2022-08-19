#   Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial

from setuptools import find_packages, setup

import paddle3d

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()


def get_all_files(directory: str):
    all_files = []
    for root, _, files in os.walk(directory):
        root = os.path.relpath(root, directory)
        for file in files:
            filepath = os.path.join(root, file)
            all_files.append(filepath)

    return all_files


def get_data_files(directory: str, data: list = None, filetypes: list = None):
    all_files = []
    data = data or []
    filetypes = filetypes or []

    for file in get_all_files(directory):
        filetype = os.path.splitext(file)[1][1:]
        filename = os.path.basename(file)
        if file in data:
            all_files.append(file)
        elif filetype in filetypes:
            all_files.append(file)

    return all_files


get_cpp_files = partial(
    get_data_files, filetypes=['h', 'hpp', 'cpp', 'cc', 'cu'])

setup(
    name='paddle3d',
    version=paddle3d.__version__.replace('-', ''),
    # TODO: add description
    description=(''),
    long_description='',
    url='https://github.com/PaddlePaddle/Paddle3D',
    author='PaddlePaddle Author',
    author_email='',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={
        'paddle3d.ops': get_cpp_files('paddle3d/ops'),
        'paddle3d.thirdparty': get_all_files('paddle3d/thirdparty')
    },
    # PyPI package information.
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=(
        'paddle3d paddlepaddle pointcloud detection classification segmentation'
    ))
