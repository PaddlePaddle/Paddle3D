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

from ..base import BaseConfig


class MonoDetConfig(BaseConfig):
    # TODO: ABC of PP3DConfig
    # Refer to https://github.com/PaddlePaddle/Paddle3D/blob/develop/paddle3d/apis/config.py
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

    def load(self, config_file_path):
        cfg_obj = Config(path=config_file_path)
        dict_ = cfg_obj.dic
        self.reset_from_dict(dict_)

    def dump(self, config_file_path):
        with open(config_file_path, 'w') as f:
            yaml.dump(self.dict, f)

    def update_dataset(self, dataset_path, dataset_type=None):
        if dataset_type is None:
            dataset_type = 'KITTI'
        if dataset_type == 'KITTI':
            ds_cfg = self._make_kitti_mono_dataset_config(dataset_path)
        else:
            raise ValueError(f"{dataset_type} is not supported.")
        self.update(ds_cfg)

    def update_optimizer(self, optimizer_type):
        # Not yet implemented
        raise NotImplementedError

    def update_backbone(self, backbone_type):
        # Not yet implemented
        raise NotImplementedError

    def update_lr_scheduler(self, lr_scheduler_type):
        # Not yet implemented
        raise NotImplementedError

    def update_batch_size(self, batch_size, mode='train'):
        # Not yet implemented
        raise NotImplementedError

    def update_weight_decay(self, weight_decay):
        # Not yet implemented
        raise NotImplementedError

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
