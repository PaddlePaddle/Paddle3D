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

import copy
import os
from typing import List, Tuple, Union

import numpy as np

from paddle3d.apis import manager
from paddle3d.datasets.waymo.waymo_det import WaymoDetDataset
from paddle3d.geometries import BBoxes3D, PointCloud
from paddle3d.sample import Sample
from paddle3d.transforms import TransformABC
from paddle3d.utils import box_utils
from paddle3d.utils.logger import logger


@manager.DATASETS.add_component
class WaymoPCDataset(WaymoDetDataset):
    def __init__(self,
                 dataset_root: str,
                 sampled_interval: int,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 class_names: Union[list, tuple] = None,
                 processed_data_tag: str = "waymo_processed_data_v1_3_2",
                 disable_nlz_flag: bool = True):
        super().__init__(
            dataset_root=dataset_root,
            sampled_interval=sampled_interval,
            mode=mode,
            transforms=transforms,
            processed_data_tag=processed_data_tag,
            class_names=class_names)
        self.disable_nlz_flag = disable_nlz_flag

    def get_lidar(self, lidar_path):
        point_features = np.load(
            lidar_path)  # (N, 6) [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if self.disable_nlz_flag:
            points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def __getitem__(self, index):
        info = copy.deepcopy(self.infos[index])
        pc_info = info["point_cloud"]
        sequence_name = pc_info["lidar_sequence"]
        sample_idx = pc_info["sample_idx"]

        lidar_path = os.path.join(self.data_path, sequence_name,
                                  "%04d.npy" % sample_idx)
        points = self.get_lidar(lidar_path)
        sample = Sample(path=lidar_path, modality='lidar')
        sample.data = PointCloud(points)

        if self.mode == "train":
            # load boxes and labels, labels starts from 0.
            gt_boxes_lidar, gt_labels, difficulties = self.load_annotation(
                index)
            sample.labels = gt_labels
            sample.difficulties = difficulties

            # TODO(liuxiao): unify coord system to avoid coord transform
            # convert boxes from [x, y, z, l, w, h, heading] to [x, y, z, w, l, h, yaw], obj_center -> bottom_center.
            # the purpose of this conversion is to reuse some data transform in paddle3d
            gt_boxes_lidar = box_utils.boxes3d_lidar_to_kitti_lidar(
                gt_boxes_lidar)
            sample.bboxes_3d = BBoxes3D(
                data=gt_boxes_lidar, coordmode=1, origin=[0.5, 0.5, 0])

        if self.transforms:
            sample = self.transforms(sample)

        return sample
