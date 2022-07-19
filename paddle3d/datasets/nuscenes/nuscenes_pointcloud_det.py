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

import os
import os.path as osp
from functools import reduce
from typing import List, Optional, Union

import numpy as np
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from paddle3d.apis import manager
from paddle3d.datasets.nuscenes.nuscenes_det import NuscenesDetDataset
from paddle3d.geometries import CoordMode
from paddle3d.sample import Sample
from paddle3d.transforms import TransformABC


@manager.DATASETS.add_component
class NuscenesPCDataset(NuscenesDetDataset):
    """
    """

    def __init__(self,
                 dataset_root: str,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 max_sweeps: int = 10,
                 class_balanced_sampling: bool = False,
                 class_names: Union[list, tuple] = None):
        super().__init__(
            dataset_root=dataset_root,
            channel="LIDAR_TOP",
            mode=mode,
            transforms=transforms,
            class_balanced_sampling=class_balanced_sampling,
            class_names=class_names)

        self.max_sweeps = max_sweeps

    def _filter(self, anno: dict, box: NuScenesBox = None) -> bool:
        # filter out objects that are not being scanned
        mask = (anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 and \
            anno['category_name'] in self.LABEL_MAP and \
            self.LABEL_MAP[anno['category_name']] in self.class_names
        return mask

    def get_sweeps(self, index: int) -> List[str]:
        """
        """
        sweeps = []
        sample = self.data[index]
        token = sample['data'][self.channel]
        sample_data = self.nusc.get('sample_data', token)

        if self.max_sweeps <= 0:
            return sweeps

        # Homogeneous transform of current sample from ego car coordinate to sensor coordinate
        curr_sample_cs = self.nusc.get("calibrated_sensor",
                                       sample_data["calibrated_sensor_token"])
        curr_sensor_from_car = transform_matrix(
            curr_sample_cs["translation"],
            Quaternion(curr_sample_cs["rotation"]),
            inverse=True)
        # Homogeneous transformation matrix of current sample from global coordinate to ego car coordinate
        curr_sample_pose = self.nusc.get("ego_pose",
                                         sample_data["ego_pose_token"])
        curr_car_from_global = transform_matrix(
            curr_sample_pose["translation"],
            Quaternion(curr_sample_pose["rotation"]),
            inverse=True,
        )
        curr_timestamp = 1e-6 * sample_data["timestamp"]

        prev_token = sample_data['prev']
        while len(sweeps) < self.max_sweeps - 1:
            if prev_token == "":
                if len(sweeps) == 0:
                    sweeps.append({
                        "lidar_path":
                        osp.join(self.dataset_root, sample_data['filename']),
                        "time_lag":
                        0,
                        "ref_from_curr":
                        None,
                    })
                else:
                    sweeps.append(sweeps[-1])
            else:
                prev_sample_data = self.nusc.get('sample_data', prev_token)
                # Homogeneous transformation matrix of previous sample from ego car coordinate to global coordinate
                prev_sample_pose = self.nusc.get(
                    "ego_pose", prev_sample_data["ego_pose_token"])
                prev_global_from_car = transform_matrix(
                    prev_sample_pose["translation"],
                    Quaternion(prev_sample_pose["rotation"]),
                    inverse=False,
                )
                # Homogeneous transform of previous sample from sensor coordinate to ego car coordinate
                prev_sample_cs = self.nusc.get(
                    "calibrated_sensor",
                    prev_sample_data["calibrated_sensor_token"])
                prev_car_from_sensor = transform_matrix(
                    prev_sample_cs["translation"],
                    Quaternion(prev_sample_cs["rotation"]),
                    inverse=False,
                )

                curr_from_pre = reduce(
                    np.dot,
                    [
                        curr_sensor_from_car, curr_car_from_global,
                        prev_global_from_car, prev_car_from_sensor
                    ],
                )
                prev_timestamp = 1e-6 * prev_sample_data["timestamp"]
                time_lag = curr_timestamp - prev_timestamp

                sweeps.append({
                    "lidar_path":
                    osp.join(self.dataset_root, prev_sample_data['filename']),
                    "time_lag":
                    time_lag,
                    "ref_from_curr":
                    curr_from_pre,
                })
                prev_token = prev_sample_data['prev']
        return sweeps

    def __getitem__(self, index: int) -> Sample:
        token = self.data[index]['data'][self.channel]
        sample_data = self.nusc.get('sample_data', token)
        path = os.path.join(self.dataset_root, sample_data['filename'])

        sample = Sample(path=path, modality="lidar")
        sample.meta.id = self.data[index]['token']
        for sweep in self.get_sweeps(index):
            sweep_sample = Sample(path=sweep["lidar_path"], modality="lidar")
            sweep_sample.meta.time_lag = sweep["time_lag"]
            sweep_sample.meta.ref_from_curr = sweep["ref_from_curr"]
            sample.sweeps.append(sweep_sample)

        if not self.is_test_mode:
            bboxes_3d, labels, attrs = self.load_annotation(index, self._filter)
            bboxes_3d.coordmode = CoordMode.NuScenesLidar
            sample.bboxes_3d = bboxes_3d
            sample.labels = labels
            sample.attrs = attrs

        if self.transforms:
            sample = self.transforms(sample)
        return sample
