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
import platform

from ..base import BaseRunner
from ..base.utils.arg import CLIArgument


class MonoDetRunner(BaseRunner):
    def train(self, config_path, cli_args, device, ips):
        python = self.distributed(device, ips)
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{python} tools/train.py --config {config_path} {args_str}"
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def evaluate(self, config_path, cli_args, device, ips):
        python = self.distributed(device, ips)
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{python} tools/evaluate.py --config {config_path} {args_str}"
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def predict(self, config_path, cli_args, device):
        raise RuntimeError(
            f"`{self.__class__.__name__}.predict()` is not implemented.")

    def export(self, config_path, cli_args, device):
        # `device` unused
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{self.python} tools/export.py --config {config_path} {args_str}"
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_path, cli_args, device, infer_dir, save_dir):
        # `config_path` and `device` unused
        args_str = ' '.join(str(arg) for arg in cli_args)
        cmd = f"{self.python} vis.py {args_str}"
        python_infer_dir = os.path.join(infer_dir, 'python')
        if save_dir is not None:
            # According to
            # https://github.com/PaddlePaddle/Paddle3D/blob/3cf884ecbc94330be0e2db780434bb60b9b4fe8c/deploy/smoke/python/vis.py#L135
            # The visualization result is saved in 'output.bmp'
            pred_path = os.path.join(python_infer_dir, 'output.bmp')
            pred_path = os.path.abspath(pred_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # TODO: Compatibility on different systems
            and_ = '&&'
            if platform.system() == 'Windows':
                mv_cmd = 'move'
            else:
                assert platform.system() in ('Linux', 'Darwin')
                mv_cmd = 'mv'
            cmd += f" {and_} {mv_cmd} {pred_path} {save_dir}"
        cp = self.run_cmd(
            cmd, switch_wdir=python_infer_dir, echo=True, silent=False)
        return cp

    def compression(self, config_path, train_cli_args, export_cli_args, device,
                    train_save_dir):
        # Step 1: Train model
        # We set `ips` to None, which disables multi-machine training.
        cp_train = self.train(config_path, train_cli_args, device, ips=None)

        # Step 2: Export model
        max_iters = None
        for arg in train_cli_args:
            if arg.key == '--iters':
                max_iters = arg.val
            elif arg.key == '--epochs':
                raise RuntimeError(
                    "Please set `--iters` instead of `--epochs`.")
        if max_iters is None:
            raise RuntimeError("`--iters` must be specified.")
        weight_path = os.path.join(train_save_dir, f"iter_{max_iters}",
                                   'model.pdparams')
        # XXX: Make in-place modification
        export_cli_args.append(CLIArgument('--model', weight_path))
        cp_export = self.export(config_path, export_cli_args, device)

        return cp_train, cp_export
