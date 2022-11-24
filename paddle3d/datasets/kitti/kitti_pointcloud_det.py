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

import numpy as np

from paddle3d.apis import manager
from paddle3d.datasets.kitti import kitti_utils
from paddle3d.datasets.kitti.kitti_det import KittiDetDataset
from paddle3d.datasets.kitti.kitti_utils import assess_object_difficulties
from paddle3d.sample import Sample


@manager.DATASETS.add_component
class KittiPCDataset(KittiDetDataset):
    """
    """

    def __getitem__(self, index: int) -> Sample:
        filename = '{}.bin'.format(self.data[index])
        path = os.path.join(self.pointcloud_dir, filename)

        sample = Sample(path=path, modality="lidar")
        sample.meta.id = self.data[index]
        calibs = self.load_calibration_info(index)
        sample["calibs"] = calibs

        if self.is_train_mode:
            kitti_records, ignored_kitti_records = self.load_annotation(index)
            difficulties = assess_object_difficulties(kitti_records)
            lidar_records = kitti_utils.project_camera_to_velodyne(
                kitti_records, calibs)
            ignored_lidar_records = kitti_utils.project_camera_to_velodyne(
                ignored_kitti_records, calibs)

            _, bboxes_3d, cls_names = kitti_utils.lidar_record_to_object(
                lidar_records)
            _, ignored_bboxes_3d, _ = kitti_utils.lidar_record_to_object(
                ignored_lidar_records)

            sample.bboxes_3d = bboxes_3d
            sample.labels = np.array(
                [self.class_names.index(name) for name in cls_names])
            sample.difficulties = difficulties
            sample.ignored_bboxes_3d = ignored_bboxes_3d
            if self.use_road_plane:
                sample.road_plane = self.load_road_plane(index)

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def load_road_plane(self, index):
        file_name = '{}.txt'.format(self.data[index])
        plane_file = os.path.join(self.base_dir, 'planes', file_name)
        if not os.path.exists(plane_file):
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @property
    def pointcloud_dir(self) -> str:
        """
        """
        return os.path.join(self.base_dir, 'velodyne')
