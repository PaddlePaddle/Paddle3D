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
import warnings

from paddle3d.datasets import KittiMonoDataset


def check_dataset(dataset_dir, dataset_type):
    if dataset_type == 'KITTI':
        # KITTI monocular 3D object detection dataset
        dataset_dir = osp.abspath(dataset_dir)
        if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
            return False
        # Construct `KITTIMonoDataset` objects via Paddle3D API
        # We do not pass in `transforms` as it is not required
        try:
            kitti_train = KittiMonoDataset(dataset_dir, mode='train')
            num_train_samples = len(kitti_train)
        except Exception as e:
            # For debugging
            warnings.warn(
                f"Exception encountered when instantiating `KittiMonoDataset` in 'train' mode. The error message is: {str(e)}"
            )
            return False
        try:
            kitti_val = KittiMonoDataset(dataset_dir, mode='val')
            num_val_samples = len(kitti_val)
        except Exception as e:
            warnings.warn(
                f"Exception encountered when instantiating `KittiMonoDataset` in 'val' mode. The error message is: {str(e)}"
            )
            return False
        # `test.txt` not supported
        return [num_train_samples, num_val_samples, None]
    else:
        raise ValueError(f"{dataset_type} is not supported.")
