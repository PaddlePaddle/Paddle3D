# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import os
from typing import List

import h5py
import numpy as np

import paddle3d.transforms as T
from paddle3d.apis import manager
from paddle3d.datasets import BaseDataset
from paddle3d.datasets.modelnet40.modelnet40_metric import AccuracyMetric
from paddle3d.geometries import PointCloud
from paddle3d.sample import Sample


@manager.DATASETS.add_component
class ModelNet40(BaseDataset):
    def __init__(self, dataset_root, num_points, transforms=None, mode='train'):
        super().__init__()
        self.data, self.label = self.load_data(dataset_root, mode)
        self.num_points = num_points
        self.mode = mode
        if isinstance(transforms, list):
            transforms = T.Compose(transforms)

        self.transforms = transforms

    def __getitem__(self, item):
        sample = Sample(path="", modality='lidar')
        sample.data = PointCloud(self.data[item][:self.num_points])
        sample.labels = self.label[item]
        if self.mode == 'train':
            if self.transforms:
                sample = self.transforms(sample)
        return sample

    def __len__(self):
        return self.data.shape[0]

    def load_data(self, dataset_root, mode):
        all_data = []
        all_label = []
        for h5_name in glob.glob(
                os.path.join(dataset_root, f"ply_data_{mode}*.h5")):
            f = h5py.File(h5_name, mode='r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    @property
    def metric(self):
        return AccuracyMetric(num_classes=40)

    @property
    def name(self) -> str:
        return "ModelNet40"

    @property
    def labels(self) -> List[str]:
        return self.label
