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

import os
import shutil

from ..base import BaseRunner
from ..base.utils.arg import CLIArgument


class MonoDetRunner(BaseRunner):

    def train(self, config_file_path, cli_args, device):
        python, device_type = self.distributed(device)
        # `device_type` ignored
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{python} tools/train.py --do_eval --config {config_file_path} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def predict(self, config_file_path, cli_args, device):
        raise RuntimeError(
            f"`{self.__class__.__name__}.predict()` is not implemented.")

    def export(self, config_file_path, cli_args, device):
        # `device` unused
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{self.python} tools/export.py --config {config_file_path} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_file_path, cli_args, device, infer_dir, save_dir):
        # `config_file_path` unused
        _, device_type = self.distributed(device)
        if device_type not in ('cpu', 'gpu'):
            raise ValueError(f"`device`={device} is not supported.")
        if device_type == 'gpu':
            cli_args.append(CLIArgument('--use_gpu', '', sep=''))
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{self.python} vis.py {args_str}"
        python_infer_dir = os.path.join(infer_dir, 'python')
        self.run_cmd(cmd,
                     switch_wdir=False,
                     wd=python_infer_dir,
                     echo=True,
                     silent=False)
        if save_dir is not None:
            # According to
            # https://github.com/PaddlePaddle/Paddle3D/blob/3cf884ecbc94330be0e2db780434bb60b9b4fe8c/deploy/smoke/python/vis.py#L135
            # The visualization result is saved in 'output.bmp'
            pred_path = os.path.join(python_infer_dir, 'output.bmp')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            shutil.move(src=pred_path, dst=save_dir)

    def compression(self, config_file_path, cli_args, device):
        python, device_type = self.distributed(device)
        # `device_type` ignored
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{python} tools/train.py --do_eval --config {config_file_path} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)
