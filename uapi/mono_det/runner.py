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
import re
from glob import iglob

from ..base import BaseRunner
from ..base.utils.arg import CLIArgument


class MonoDetRunner(BaseRunner):
    def train(self, config_path, cli_args, device):
        python, device_type = self.distributed(device)
        # `device_type` ignored
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{python} tools/train.py --do_eval --config {config_path} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def predict(self, config_path, cli_args, device):
        raise RuntimeError(
            f"`{self.__class__.__name__}.predict()` is not implemented.")

    def export(self, config_path, cli_args, device):
        # `device` unused
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{self.python} tools/export.py --config {config_path} {args_str}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_path, cli_args, device, infer_dir, save_dir):
        # `config_path` unused
        _, device_type = self.distributed(device)
        if device_type not in ('cpu', 'gpu'):
            raise ValueError(f"`device`={device} is not supported.")
        if device_type == 'gpu':
            cli_args.append(CLIArgument('--use_gpu', '', sep=''))
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{self.python} vis.py {args_str}"
        python_infer_dir = os.path.join(infer_dir, 'python')
        self.run_cmd(cmd, switch_wdir=python_infer_dir, echo=True, silent=False)
        if save_dir is not None:
            # According to
            # https://github.com/PaddlePaddle/Paddle3D/blob/3cf884ecbc94330be0e2db780434bb60b9b4fe8c/deploy/smoke/python/vis.py#L135
            # The visualization result is saved in 'output.bmp'
            pred_path = os.path.join(python_infer_dir, 'output.bmp')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            shutil.move(src=pred_path, dst=save_dir)

    def compression(self, config_path, train_cli_args, export_cli_args, device,
                    train_save_dir):
        # Step 1: Train model
        self.train(config_path, train_cli_args, device)

        # Step 2: Export model
        # Get the path of all checkpoints and find the latest one
        max_iter = 0
        latest_ckp_path = None
        for p in iglob(os.path.join(train_save_dir, 'iter_*')):
            m = re.search(r'iter_(\d+)$', p)
            if m is not None:
                iter_ = m.group(1)
                iter_ = int(iter_)
                if iter_ > max_iter:
                    max_iter = iter_
                    latest_ckp_path = p
        if not os.path.exists(latest_ckp_path):
            raise FileNotFoundError
        # XXX: Make in-place modification
        export_cli_args.append(
            CLIArgument('--model',
                        os.path.join(latest_ckp_path, 'model.pdparams')))
        self.export(config_path, export_cli_args, device)
