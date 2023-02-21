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

from ..pp3d_config import PP3DConfig


class MonoDetConfig(PP3DConfig):
    def update_dataset(self, dataset_dir, dataset_type=None):
        if dataset_type is None:
            dataset_type = 'KITTI'
        if dataset_type == 'KITTI':
            ds_cfg = self._make_kitti_mono_dataset_config(dataset_dir)
        else:
            raise ValueError(f"{dataset_type} is not supported.")
        # Prune old config
        keys_to_keep = ('transforms', 'mode')
        if 'train_dataset' in self:
            for key in list(
                    k for k in self.train_dataset if k not in keys_to_keep):
                self.train_dataset.pop(key)
        if 'val_dataset' in self:
            for key in list(
                    k for k in self.val_dataset if k not in keys_to_keep):
                self.val_dataset.pop(key)
        self.update(ds_cfg)

    def _update_amp(self, amp):
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
        self.update(amp_cfg)
