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

import yaml
from paddle3d.apis.config import Config

from .base import BaseConfig


class PP3DConfig(BaseConfig):
    # Refer to https://github.com/PaddlePaddle/Paddle3D/blob/release/1.0/paddle3d/apis/config.py
    def update(self, dict_like_obj):
        def _merge_config_dicts(dict_from, dict_to):
            # According to
            # https://github.com/PaddlePaddle/Paddle3D/blob/3cf884ecbc94330be0e2db780434bb60b9b4fe8c/paddle3d/apis/config.py#L90
            for key, val in dict_from.items():
                if isinstance(val, dict) and key in dict_to:
                    dict_to[key] = _merge_config_dicts(val, dict_to[key])
                else:
                    dict_to[key] = val
            return dict_to

        dict_ = _merge_config_dicts(dict_like_obj, self.dict)
        self.reset_from_dict(dict_)

    def load(self, config_path):
        cfg_obj = Config(path=config_path)
        dict_ = cfg_obj.dic
        self.reset_from_dict(dict_)

    def dump(self, config_path):
        with open(config_path, 'w') as f:
            yaml.dump(self.dict, f)

    def update_learning_rate(self, learning_rate):
        if 'lr_scheduler' not in self:
            raise RuntimeError(
                "Not able to update learning rate, because no LR scheduler config was found."
            )
        self.lr_scheduler['learning_rate'] = learning_rate

    def update_batch_size(self, batch_size, mode='train'):
        if mode == 'train':
            self.set_val('batch_size', batch_size)
        else:
            raise ValueError(
                f"Setting `batch_size` in '{mode}' mode is not supported.")

    def _make_kitti_mono_dataset_config(self, dataset_root_path):
        return {
            'train_dataset': {
                'type': 'KittiMonoDataset',
                'dataset_root': dataset_root_path,
            },
            'val_dataset': {
                'type': 'KittiMonoDataset',
                'dataset_root': dataset_root_path,
            },
        }

    def _get_epochs_iters(self):
        # TODO
        raise NotImplementedError

    def _get_learning_rate(self):
        # TODO
        raise NotImplementedError

    def _get_batch_size(self, mode='train'):
        # TODO
        raise NotImplementedError

    def _get_qat_epochs_iters(self):
        # TODO
        raise NotImplementedError

    def _get_qat_learning_rate(self):
        # TODO
        raise NotImplementedError
