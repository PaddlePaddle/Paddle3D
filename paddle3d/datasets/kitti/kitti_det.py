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

import csv
import os
from typing import List, Tuple, Union, Dict

import numpy as np
import pandas

from paddle3d import transforms as T
from paddle3d.datasets import BaseDataset
from paddle3d.datasets.kitti.kitti_metric import KittiMetric
from paddle3d.transforms import TransformABC


class KittiDetDataset(BaseDataset):
    """
    """

    def __init__(self,
                 dataset_root: str,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 class_names: Union[list, tuple] = None,
                 CLASS_MAP: Dict[str, int] = None,
                 class_balanced_sampling: bool = False,
                 use_road_plane: bool = False):
        super().__init__()
        self.dataset_root = dataset_root
        self.mode = mode.lower()

        if isinstance(transforms, list):
            transforms = T.Compose(transforms)

        self.transforms = transforms
        self.class_names = class_names
        self.use_road_plane = use_road_plane
        if CLASS_MAP is None:
            self.CLASS_MAP = {'Car': 0, 'Cyclist': 1, 'Pedestrian': 2}
        else:
            self.CLASS_MAP = CLASS_MAP
        self.CLASS_MAP_REVERSE = {
            value: key
            for key, value in self.CLASS_MAP.items()
        }
        if self.class_names is None:
            self.class_names = list(self.CLASS_MAP.keys())

        if self.mode not in ['train', 'val', 'trainval', 'test']:
            raise ValueError(
                "mode should be 'train', 'val', 'trainval' or 'test', but got {}."
                .format(self.mode))

        # get file list
        with open(self.imagesets_path) as file:
            self.data = file.read().strip('\n').split('\n')

        if class_balanced_sampling and self.mode.lower() == 'train' and len(
                self.class_names) > 1:
            cls_dist = {class_name: [] for class_name in self.class_names}
            for index in range(len(self.data)):
                file_idx = self.data[index]
                kitti_records, ignored_kitti_records = self.load_annotation(
                    index)
                gt_names = []
                for anno in kitti_records:
                    class_name = anno[0]
                    if class_name in self.class_names:
                        gt_names.append(class_name)
                for class_name in set(gt_names):
                    cls_dist[class_name].append(file_idx)

            num_balanced_samples = sum([len(v) for k, v in cls_dist.items()])
            num_balanced_samples = max(num_balanced_samples, 1)
            balanced_frac = 1.0 / len(self.class_names)
            fracs = [len(v) / num_balanced_samples for k, v in cls_dist.items()]
            sampling_ratios = [balanced_frac / frac for frac in fracs]

            resampling_data = []
            for samples, sampling_ratio in zip(
                    list(cls_dist.values()), sampling_ratios):
                resampling_data.extend(samples)
                if sampling_ratio > 1.:
                    resampling_data.extend(
                        np.random.choice(
                            samples,
                            int(len(samples) * (sampling_ratio - 1.))).tolist())
            self.data = resampling_data
        self.use_road_plane = use_road_plane

    def __len__(self):
        return len(self.data)

    @property
    def base_dir(self) -> str:
        """
        """
        dirname = 'testing' if self.is_test_mode else 'training'
        return os.path.join(self.dataset_root, dirname)

    @property
    def label_dir(self) -> str:
        """
        """
        return os.path.join(self.base_dir, 'label_2')

    @property
    def calib_dir(self) -> str:
        """
        """
        return os.path.join(self.base_dir, 'calib')

    @property
    def imagesets_path(self) -> str:
        """
        """
        return os.path.join(self.dataset_root, 'ImageSets',
                            '{}.txt'.format(self.mode))

    def load_calibration_info(self, index: int, use_data: bool = True) -> Tuple:
        """
        """
        if use_data:
            filename = '{}.txt'.format(self.data[index])
        else:
            filename = '{}.txt'.format(index)

        with open(os.path.join(self.calib_dir, filename), 'r') as csv_file:
            reader = list(csv.reader(csv_file, delimiter=' '))

            # parse camera intrinsics from calibration table
            P0 = [float(i) for i in reader[0][1:]]
            P0 = np.array(P0, dtype=np.float32).reshape(3, 4)

            P1 = [float(i) for i in reader[1][1:]]
            P1 = np.array(P1, dtype=np.float32).reshape(3, 4)

            P2 = [float(i) for i in reader[2][1:]]
            P2 = np.array(P2, dtype=np.float32).reshape(3, 4)

            P3 = [float(i) for i in reader[3][1:]]
            P3 = np.array(P3, dtype=np.float32).reshape(3, 4)

            # parse correction matrix for camera 0.
            R0_rect = [float(i) for i in reader[4][1:]]
            R0_rect = np.array(R0_rect, dtype=np.float32).reshape(3, 3)

            # parse matrix from velodyne to camera
            V2C = [float(i) for i in reader[5][1:]]
            V2C = np.array(V2C, dtype=np.float32).reshape(3, 4)

            if len(reader) == 6:
                # parse matrix from imu to velodyne
                I2V = [float(i) for i in reader[6][1:]]
                I2V = np.array(I2V, dtype=np.float32).reshape(3, 4)
            else:
                I2V = np.array([0, 4], dtype=np.float32)

        return P0, P1, P2, P3, R0_rect, V2C, I2V

    def load_annotation(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        filename = '{}.txt'.format(self.data[index])
        with open(os.path.join(self.label_dir, filename), 'r') as csv_file:
            df = pandas.read_csv(csv_file, sep=' ', header=None)
            array = np.array(df)
            rows = []
            ignored_rows = []
            for row in array:
                if row[0] in self.class_names:
                    rows.append(row)
                elif row[0] != 'DontCare':
                    ignored_rows.append(row)

        kitti_records = np.array(rows)
        ignored_kitti_records = np.array(ignored_rows)
        return kitti_records, ignored_kitti_records

    @property
    def metric(self):
        gt = []
        for idx in range(len(self)):
            annos = self.load_annotation(idx)
            if len(annos[0]) > 0 and len(annos[1]) > 0:
                gt.append(np.concatenate((annos[0], annos[1]), axis=0))
            elif len(annos[0]) > 0:
                gt.append(annos[0])
            else:
                gt.append(annos[1])
        return KittiMetric(
            groundtruths=gt,
            classmap={i: name
                      for i, name in enumerate(self.class_names)},
            indexes=self.data)

    @property
    def name(self) -> str:
        return "KITTI"

    @property
    def labels(self) -> List[str]:
        return self.class_names
