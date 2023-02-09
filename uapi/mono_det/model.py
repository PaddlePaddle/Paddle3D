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

from ..base import BaseModel
from ..base.utils.arg import CLIArgument
from ..base.utils.misc import abspath
from .config import MonoDetConfig


class MonoDetModel(BaseModel):
    _SAVE_NAME = 'model'

    def train(self,
              dataset,
              batch_size=None,
              learning_rate=None,
              epochs_iters=None,
              device=None,
              resume_path=None,
              dy2st=None,
              amp=None,
              save_dir=None):
        # NOTE: We must use an absolute path here,
        # so we can run the scripts either inside or outside the repo dir.
        dataset = abspath(dataset)
        if dy2st == True:
            raise ValueError(f"`dy2st`={dy2st} is not supported.")
        if resume_path is not None:
            resume_path = abspath(resume_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.model_info['config_path']
        config = MonoDetConfig.build_from_file(config_file_path)
        config._update_dataset_config(dataset)
        if amp is not None:
            # XXX: Currently, we hard-code the AMP settings according to
            # https://github.com/PaddlePaddle/Paddle3D/blob/3cf884ecbc94330be0e2db780434bb60b9b4fe8c/configs/smoke/smoke_dla34_no_dcn_kitti_amp.yml#L6
            amp_cfg = {
                'amp_cfg': {
                    'enable': True,
                    'level': amp,
                    'scaler': {
                        'init_loss_scaling': 1024.0
                    },
                    'custom_black_list': ['matmul_v2', 'elementwise_mul']
                }
            }
            config.update(amp_cfg)
        config_file_path = self.config_file_path
        config.dump(config_file_path)

        # Parse CLI arguments
        cli_args = []
        if batch_size is not None:
            cli_args.append(CLIArgument('--batch_size', batch_size))
        if learning_rate is not None:
            cli_args.append(CLIArgument('--learning_rate', learning_rate))
        if epochs_iters is not None:
            # NOTE: Paddle3D supports both `--iters` and `--epochs`, and we use `--iters` here.
            cli_args.append(CLIArgument('--iters', epochs_iters))
        if resume_path is not None:
            if save_dir is not None:
                raise ValueError(
                    "When `resume_path` is not None, `save_dir` must be set to None."
                )
            model_dir = osp.dirname(resume_path)
            cli_args.append(CLIArgument('--resume', '', sep=''))
            cli_args.append(CLIArgument('--save_dir', model_dir))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        self.runner.train(config_file_path, cli_args, device)

    def predict(self,
                weight_path=None,
                device=None,
                input_path=None,
                save_dir=None):
        raise RuntimeError(
            f"`{self.__class__.__name__}.predict()` is not implemented.")

    def export(self, weight_path=None, save_dir=None):
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.model_info['config_path']
        config = MonoDetConfig.build_from_file(config_file_path)
        config_file_path = self.config_file_path
        config.dump(config_file_path)

        # Parse CLI arguments
        cli_args = []
        if weight_path is not None:
            cli_args.append(CLIArgument('--model', weight_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))
        # Fix save name
        cli_args.append(CLIArgument('--save_name', self._SAVE_NAME))

        self.runner.export(config_file_path, cli_args, None)

    def infer(self, model_dir, device=None, input_path=None, save_dir=None):
        model_dir = abspath(model_dir)
        if input_path is not None:
            input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Parse CLI arguments
        cli_args = []
        model_file_path = osp.join(model_dir, f'{self._SAVE_NAME}.pdmodel')
        params_file_path = osp.join(model_dir, f'{self._SAVE_NAME}.pdiparams')
        cli_args.append(CLIArgument('--model_file', model_file_path))
        cli_args.append(CLIArgument('--params_file', params_file_path))
        if input_path is not None:
            cli_args.append(CLIArgument('--image', input_path))

        infer_dir = self.model_info['infer_dir']
        # The inference script does not require a config file
        self.runner.infer(None, cli_args, device, infer_dir, save_dir)

    def compression(self,
                    dataset,
                    batch_size=None,
                    learning_rate=None,
                    epochs_iters=None,
                    device=None,
                    weight_path=None,
                    save_dir=None):
        dataset = abspath(dataset)
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.model_info['config_path']
        config = MonoDetConfig.build_from_file(config_file_path)
        config._update_dataset_config(dataset)
        config_file_path = self.config_file_path
        config.dump(config_file_path)
        ac_config_file_path = self.model_info['auto_compression_config_path']
        # TODO: Allow updates of auto compression config file

        # Parse CLI arguments
        cli_args = []
        cli_args.append(CLIArgument('--quant_config', ac_config_file_path))
        if batch_size is not None:
            cli_args.append(CLIArgument('--batch_size', batch_size))
        if learning_rate is not None:
            cli_args.append(CLIArgument('--learning_rate', learning_rate))
        if epochs_iters is not None:
            # NOTE: Paddle3D supports both `--iters` and `--epochs`, and we use `--iters` here.
            cli_args.append(CLIArgument('--iters', epochs_iters))
        if weight_path is not None:
            cli_args.append(CLIArgument('--model', weight_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        self.runner.compression(config_file_path, cli_args, device)
