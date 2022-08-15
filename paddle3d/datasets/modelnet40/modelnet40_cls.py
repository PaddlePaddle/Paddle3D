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

import h5py
import numpy as np

from paddle3d.apis import manager
from paddle3d.datasets import BaseDataset
from paddle3d.datasets.modelnet40.modelnet40_metric import ModelNet40Metric


@manager.DATASETS.add_component
class ModelNet40(BaseDataset):
    def __init__(self,
                 dataset_root,
                 num_points,
                 transforms=None,
                 pt_norm=False,
                 mode='train'):
        super().__init__()
        self.data, self.label = self.load_data(dataset_root, mode)
        self.num_points = num_points
        self.mode = mode
        self.pt_norm = pt_norm
        self.transforms = transforms

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.mode == 'train':
            if self.pt_norm:
                pointcloud = self.pc_normalize(pointcloud)
            for transform in self.transforms:
                pointcloud = transform(pointcloud)
            np.random.shuffle(pointcloud)  # shuffle the order of pts
        return {'data': pointcloud, 'labels': label}

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

    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    @property
    def metric(self):
        return ModelNet40Metric(num_classes=40)
