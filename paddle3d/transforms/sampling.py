# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

__all__ = ["SamplingDatabase"]

import os.path as osp
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np

from paddle3d.apis import manager
from paddle3d.geometries.bbox import BBoxes3D, box_collision_test
from paddle3d.geometries.pointcloud import PointCloud
from paddle3d.sample import Sample
from paddle3d.transforms.base import TransformABC
from paddle3d.utils.logger import logger


@manager.TRANSFORMS.add_component
class SamplingDatabase(TransformABC):
    """
    Sample objects from ground truth database and paste on current scene.

    Args:
        min_num_points_in_box_per_class (Dict[str, int]): Minimum number of points in sampled object for each class.
        max_num_samples_per_class (Dict[str, int]): Maximum number of objects sampled from each class.
        database_anno_path (str): Path to database annotation file (.pkl).
        database_root (str): Path to database root directory.
        class_names (List[str]): List of class names.
        ignored_difficulty (List[int]): List of difficulty levels to be ignored.
    """

    def __init__(self,
                 min_num_points_in_box_per_class: Dict[str, int],
                 max_num_samples_per_class: Dict[str, int],
                 database_anno_path: str,
                 database_root: str,
                 class_names: List[str],
                 ignored_difficulty: List[int] = None):
        self.min_num_points_in_box_per_class = min_num_points_in_box_per_class
        self.max_num_samples_per_class = max_num_samples_per_class
        self.database_anno_path = database_anno_path
        with open(database_anno_path, "rb") as f:
            database_anno = pickle.load(f)
        if not osp.exists(database_root):
            raise ValueError(
                f"Database root path {database_root} does not exist!!!")
        self.database_root = database_root
        self.class_names = class_names
        self.database_anno = self._filter_min_num_points_in_box(database_anno)
        self.ignored_difficulty = ignored_difficulty
        if ignored_difficulty is not None:
            self.database_anno = self._filter_ignored_difficulty(
                self.database_anno)

        self.sampler_per_class = dict()
        for cls_name, annos in self.database_anno.items():
            self.sampler_per_class[cls_name] = Sampler(cls_name, annos)

    def _filter_min_num_points_in_box(self, database_anno: Dict[str, list]):
        new_database_anno = defaultdict(list)
        for cls_name, annos in database_anno.items():
            if cls_name not in self.class_names or cls_name not in self.min_num_points_in_box_per_class:
                continue
            logger.info("Load {} {} database infos".format(
                len(annos), cls_name))
            for anno in annos:
                if anno["num_points_in_box"] >= self.min_num_points_in_box_per_class[
                        cls_name]:
                    new_database_anno[cls_name].append(anno)
        logger.info("After filtering min_num_points_in_box:")
        for cls_name, annos in new_database_anno.items():
            logger.info("Load {} {} database infos".format(
                len(annos), cls_name))
        return new_database_anno

    def _filter_ignored_difficulty(self, database_anno: Dict[str, list]):
        new_database_anno = defaultdict(list)
        for cls_name, annos in database_anno.items():
            if cls_name not in self.class_names or cls_name not in self.min_num_points_in_box_per_class:
                continue
            for anno in annos:
                if anno["difficulty"] not in self.ignored_difficulty:
                    new_database_anno[cls_name].append(anno)
        logger.info("After filtering ignored difficulty:")
        for cls_name, annos in new_database_anno.items():
            logger.info("Load {} {} database infos".format(
                len(annos), cls_name))
        return new_database_anno

    def _convert_box_format(self, bboxes_3d):
        # convert to [x,y,z,l,w,h,heading], original is [x,y,z,w,l,h,yaw]
        bboxes_3d[:, 2] += bboxes_3d[:, 5] / 2
        bboxes_3d[:, 3:6] = bboxes_3d[:, [4, 3, 5]]
        bboxes_3d[:, 6] = -(bboxes_3d[:, 6] + np.pi / 2)
        return bboxes_3d

    def _convert_box_format_back(self, bboxes_3d):
        bboxes_3d[:, 2] -= bboxes_3d[:, 5] / 2
        bboxes_3d[:, 3:6] = bboxes_3d[:, [4, 3, 5]]
        bboxes_3d[:, 6] = -(bboxes_3d[:, 6] + np.pi / 2)
        return bboxes_3d

    def _lidar_to_rect(self, pts_lidar, R0, V2C):
        pts_lidar_hom = self._cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(V2C.T, R0.T))
        return pts_rect

    def _rect_to_lidar(self, pts_rect, R0, V2C):
        pts_rect_hom = self._cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4),
                                             dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1
        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(
            np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def _cart_to_hom(self, pts):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def _put_boxes_on_road_planes(self, sampled_boxes, road_planes, calibs):
        a, b, c, d = road_planes
        R0, V2C = calibs[4], calibs[5]
        sampled_boxes = self._convert_box_format(sampled_boxes)
        center_cam = self._lidar_to_rect(sampled_boxes[:, 0:3], R0, V2C)
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = self._rect_to_lidar(center_cam, R0, V2C)[:, 2]
        mv_height = sampled_boxes[:,
                                  2] - sampled_boxes[:, 5] / 2 - cur_lidar_height
        sampled_boxes[:, 2] -= mv_height
        sampled_boxes = self._convert_box_format_back(sampled_boxes)
        return sampled_boxes, mv_height

    def sampling(self, sample: Sample, num_samples_per_class: Dict[str, int]):
        existing_bboxes_3d = sample.bboxes_3d.copy()
        existing_velocities = None
        if sample.bboxes_3d.velocities is not None:
            existing_velocities = sample.bboxes_3d.velocities.copy()
        existing_labels = sample.labels.copy()
        existing_data = sample.data.copy()
        existing_difficulties = getattr(sample, "difficulties", None)
        ignored_bboxes_3d = getattr(
            sample, "ignored_bboxes_3d",
            np.zeros([0, existing_bboxes_3d.shape[1]],
                     dtype=existing_bboxes_3d.dtype))
        avoid_coll_bboxes_3d = np.vstack(
            [existing_bboxes_3d, ignored_bboxes_3d])

        for cls_name, num_samples in num_samples_per_class.items():
            if num_samples > 0:
                sampling_annos = self.sampler_per_class[cls_name].sampling(
                    num_samples)
                num_sampling = len(sampling_annos)
                indices = np.arange(num_sampling)
                sampling_bboxes_3d = np.vstack(
                    [sampling_annos[i]["bbox_3d"] for i in range(num_sampling)])

                sampling_bboxes = BBoxes3D(
                    sampling_bboxes_3d,
                    coordmode=sample.bboxes_3d.coordmode,
                    origin=sample.bboxes_3d.origin)
                avoid_coll_bboxes = BBoxes3D(
                    avoid_coll_bboxes_3d,
                    coordmode=sample.bboxes_3d.coordmode,
                    origin=sample.bboxes_3d.origin)
                s_bboxes_bev = sampling_bboxes.corners_2d
                e_bboxes_bev = avoid_coll_bboxes.corners_2d
                # filter the sampling bboxes which cross over the existing bboxes
                total_bv = np.concatenate([e_bboxes_bev, s_bboxes_bev], axis=0)
                coll_mat = box_collision_test(total_bv, total_bv)
                diag = np.arange(total_bv.shape[0])
                coll_mat[diag, diag] = False
                idx = e_bboxes_bev.shape[0]
                mask = []
                for num in range(num_sampling):
                    if coll_mat[idx + num].any():
                        coll_mat[idx + num] = False
                        coll_mat[:, idx + num] = False
                        mask.append(False)
                    else:
                        mask.append(True)
                indices = indices[mask]

                # put all boxes(without filter) on road plane
                sampling_bboxes_3d_copy = sampling_bboxes_3d.copy()
                if hasattr(sample, "road_plane"):
                    sampling_bboxes_3d, mv_height = self._put_boxes_on_road_planes(
                        sampling_bboxes_3d, sample.road_plane, sample.calibs)

                if len(indices) > 0:
                    sampling_data = []
                    sampling_labels = []
                    sampling_velocities = []
                    sampling_difficulties = []
                    label = self.class_names.index(cls_name)
                    for i in indices:
                        if existing_velocities is not None:
                            sampling_velocities.append(
                                sampling_annos[i]["velocity"])
                        if existing_difficulties is not None:
                            sampling_difficulties.append(
                                sampling_annos[i]["difficulty"])

                        sampling_labels.append(label)
                        lidar_data = np.fromfile(
                            osp.join(self.database_root,
                                     sampling_annos[i]["lidar_file"]),
                            "float32").reshape(
                                [-1, sampling_annos[i]["lidar_dim"]])
                        lidar_data[:, 0:3] += sampling_bboxes_3d_copy[i, 0:3]
                        if hasattr(sample, "road_plane"):
                            lidar_data[:, 2] -= mv_height[i]
                        sampling_data.append(lidar_data)

                    existing_bboxes_3d = np.vstack(
                        [existing_bboxes_3d, sampling_bboxes_3d[indices]])
                    avoid_coll_bboxes_3d = np.vstack(
                        [avoid_coll_bboxes_3d, sampling_bboxes_3d[indices]])
                    if sample.bboxes_3d.velocities is not None:
                        existing_velocities = np.vstack(
                            [existing_velocities, sampling_velocities])
                    existing_labels = np.hstack(
                        [existing_labels, sampling_labels])
                    existing_data = np.vstack(
                        [np.vstack(sampling_data), existing_data])
                    if existing_difficulties is not None:
                        existing_difficulties = np.hstack(
                            [existing_difficulties, sampling_difficulties])

        result = {
            "bboxes_3d": existing_bboxes_3d,
            "data": existing_data,
            "labels": existing_labels
        }
        if existing_velocities is not None:
            result.update({"velocities": existing_velocities})
        if existing_difficulties is not None:
            result.update({"difficulties": existing_difficulties})
        return result

    def _cal_num_samples_per_class(self, sample: Sample):
        labels = sample.labels
        num_samples_per_class = dict()
        for cls_name, max_num_samples in self.max_num_samples_per_class.items():
            label = self.class_names.index(cls_name)
            if label in labels:
                num_existing = np.sum([int(label) == int(l) for l in labels])
                num_samples = 0 if num_existing > max_num_samples else max_num_samples - num_existing
                num_samples_per_class[cls_name] = num_samples
            else:
                num_samples_per_class[cls_name] = max_num_samples
        return num_samples_per_class

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError(
                "Sampling from a database only supports lidar data!")

        num_samples_per_class = self._cal_num_samples_per_class(sample)
        samples = self.sampling(sample, num_samples_per_class)

        sample.bboxes_3d = BBoxes3D(
            samples["bboxes_3d"],
            coordmode=sample.bboxes_3d.coordmode,
            origin=sample.bboxes_3d.origin)
        sample.labels = samples["labels"]
        if "velocities" in samples:
            sample.bboxes_3d.velocities = samples["velocities"]
        if "difficulties" in samples:
            sample.difficulties = samples["difficulties"]
        sample.data = PointCloud(samples["data"])
        return sample


class Sampler(object):
    def __init__(self, cls_name: str, annos: List[dict], shuffle: bool = True):
        self.shuffle = shuffle
        self.cls_name = cls_name
        self.annos = annos
        self.idx = 0
        self.length = len(annos)
        self.indices = np.arange(len(annos))
        if shuffle:
            np.random.shuffle(self.indices)

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.idx = 0

    def sampling(self, num_samples):
        if self.idx + num_samples >= self.length:
            indices = self.indices[self.idx:].copy()
            self.reset()
        else:
            indices = self.indices[self.idx:self.idx + num_samples]
            self.idx += num_samples

        sampling_annos = [self.annos[i] for i in indices]
        return sampling_annos
