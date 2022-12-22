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
import pickle
from typing import Callable, List, Tuple, Union

import numpy as np
import paddle

import paddle3d.transforms as T
from paddle3d.datasets import BaseDataset
from paddle3d.geometries import BBoxes2D, BBoxes3D
from paddle3d.sample import Sample
from paddle3d.transforms import TransformABC
from paddle3d.utils.logger import logger


class WaymoDetDataset(BaseDataset):
    def __init__(self,
                 dataset_root: str,
                 sampled_interval: int,
                 mode: str = "train",
                 transforms: Union[TransformABC, List[TransformABC]] = None,
                 class_names: Union[list, tuple] = None,
                 processed_data_tag: str = "waymo_processed_data_v1_3_2"):
        super().__init__()
        self.dataset_root = dataset_root
        self.data_path = os.path.join(self.dataset_root, processed_data_tag)
        self.sampled_interval = sampled_interval
        self.mode = mode.lower()

        if isinstance(transforms, list):
            transforms = T.Compose(transforms)
        self.transforms = transforms
        self.class_names = class_names

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))
        split_dir = os.path.join(self.dataset_root, "ImageSets",
                                 self.mode + '.txt')
        self.sample_sequence_list = [
            x.strip() for x in open(split_dir).readlines()
        ]

        self.infos = []
        self.load_waymo_infos()

    def load_waymo_infos(self):
        logger.info("Loading Waymo Dataset")
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = os.path.join(self.data_path, sequence_name,
                                     "{}.pkl".format(sequence_name))
            if not os.path.exists(info_path):
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        self.infos = waymo_infos
        logger.info("Total skipped sequences {}".format(num_skipped_infos))
        logger.info("Total samples for Waymo dataset: {}".format(
            len(waymo_infos)))

        if self.sampled_interval > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.sampled_interval):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            logger.info("Total sampled samples for Waymo dataset: {}".format(
                len(self.infos)))

    def drop_info_with_name(self, info, name):
        ret_info = {}
        keep_indices = [i for i, x in enumerate(info['name']) if x != name]
        for key in info.keys():
            ret_info[key] = info[key][keep_indices]
        return ret_info

    def __len__(self):
        return len(self.infos)

    def load_annotation(self, index):
        info = copy.deepcopy(self.infos[index])
        annos = info["annos"]
        # filter unknown class
        annos = self.drop_info_with_name(annos, name='unknown')

        # filter empty boxes for train
        gt_boxes_lidar = annos['gt_boxes_lidar']
        difficulty = annos['difficulty']
        mask = (annos['num_points_in_gt'] > 0)
        gt_names = annos['name'][mask]
        gt_boxes_lidar = gt_boxes_lidar[mask]
        difficulty = difficulty[mask]

        # filter boxes with given classes
        mask = [i for i, x in enumerate(gt_names) if x in self.class_names]
        mask = np.array(mask, dtype=np.int64)
        gt_names = gt_names[mask]
        gt_boxes_lidar = gt_boxes_lidar[mask]
        difficulty = difficulty[mask]
        gt_labels = np.array([self.class_names.index(n) for n in gt_names],
                             dtype=np.int32)

        return gt_boxes_lidar, gt_labels, difficulty

    @property
    def metric(self):
        # lazy import to avoid tensorflow dependency in other tasks
        from paddle3d.datasets.waymo.waymo_metric import WaymoMetric
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]
        return WaymoMetric(
            eval_gt_annos, self.class_names, distance_thresh=1000)

    @property
    def name(self) -> str:
        return "Waymo"

    @property
    def labels(self) -> List[str]:
        return self.class_names
