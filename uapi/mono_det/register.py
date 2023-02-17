# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os.path as osp

from ..base.register import register_model_info, register_suite_info
from .model import MonoDetModel
from .runner import MonoDetRunner
from ..config import PP3DConfig

# XXX: Hard-code relative path of repo root dir
_file_path = osp.realpath(__file__)
REPO_ROOT_PATH = osp.abspath(osp.join(osp.dirname(_file_path), '..', '..'))
register_suite_info({
    'suite_name': 'MonoDet',
    'model': MonoDetModel,
    'runner': MonoDetRunner,
    'config': PP3DConfig,
    'runner_root_path': REPO_ROOT_PATH
})

register_model_info({
    'model_name':
    'smoke',
    'suite':
    'MonoDet',
    'config_path':
    osp.join(REPO_ROOT_PATH, 'configs', 'smoke',
             'smoke_dla34_no_dcn_kitti.yml'),
    'auto_compression_config_path':
    osp.join(REPO_ROOT_PATH, 'configs', 'quant', 'smoke_kitti.yml'),
    'supported_apis': ['train', 'export', 'infer', 'compression'],
    # Additional info
    'infer_dir':
    'deploy/smoke'
})
