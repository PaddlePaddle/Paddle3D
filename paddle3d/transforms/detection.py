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

import os.path as osp
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from paddle3d.apis import manager
from paddle3d.geometries.bbox import (BBoxes3D, box_collision_test,
                                      points_in_convex_polygon_3d_jit,
                                      rbbox2d_to_near_bbox)
from paddle3d.geometries.pointcloud import PointCloud
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
from paddle3d.transforms.functional import points_to_voxel
from paddle3d.utils.logger import logger

__all__ = [
    "Normalize", "RandomHorizontalFlip", "RandomVerticalFlip", "GlobalRotate",
    "GlobalScale", "GlobalTranslate", "ShufflePoint", "FilterBBoxOutsideRange",
    "HardVoxelize", "SamplingDatabase", "RandomObjectPerturb", "GenerateAnchors"
]


@manager.TRANSFORMS.add_component
class Normalize(TransformABC):
    """
    """

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]):
        self.mean = mean
        self.std = std

        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))

        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample: Sample):
        """
        """
        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
        std = np.array(self.std)[:, np.newaxis, np.newaxis]

        if sample.modality == 'image':
            sample.data = sample.data.astype(np.float32, copy=False) / 255.0

            if sample.meta.channel_order != 'chw':
                mean = np.array(self.mean)
                std = np.array(self.std)

        sample.data = F.normalize(sample.data, mean, std)
        return sample


@manager.TRANSFORMS.add_component
class RandomHorizontalFlip(TransformABC):
    """
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample: Sample):
        if np.random.random() < self.prob:
            if sample.modality == "image":
                sample.data = F.horizontal_flip(sample.data)
                h, w, c = sample.data.shape
            elif sample.modality == "lidar":
                sample.data.flip(axis=1)

            # Flip camera intrinsics
            if "camera_intrinsic" in sample.meta:
                sample.meta.camera_intrinsic[
                    0, 2] = w - sample.meta.camera_intrinsic[0, 2] - 1

            # Flip bbox
            if sample.bboxes_3d is not None:
                sample.bboxes_3d.horizontal_flip()
            if sample.bboxes_2d is not None and sample.modality == "image":
                sample.bboxes_2d.horizontal_flip(image_width=w)
        return sample


@manager.TRANSFORMS.add_component
class RandomVerticalFlip(TransformABC):
    """
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample: Sample):
        if np.random.random() < self.prob:
            if sample.modality == "image":
                sample.data = F.vertical_flip(sample.data)
                h, w, c = sample.data.shape
            elif sample.modality == "lidar":
                sample.data.flip(axis=0)

            # Flip camera intrinsics
            if "camera_intrinsic" in sample.meta:
                sample.meta.camera_intrinsic[
                    1, 2] = h - sample.meta.camera_intrinsic[1, 2] - 1

            # Flip bbox
            if sample.bboxes_3d is not None:
                sample.bboxes_3d.vertical_flip()
            if sample.bboxes_2d is not None and sample.modality == "image":
                sample.bboxes_2d.vertical_flip(image_height=h)

        return sample


@manager.TRANSFORMS.add_component
class GlobalRotate(TransformABC):
    """
    """

    def __init__(self, min_rot: float = -np.pi / 4, max_rot: float = np.pi / 4):
        self.min_rot = min_rot
        self.max_rot = max_rot

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalRotate only supports lidar data!")
        angle = np.random.uniform(self.min_rot, self.max_rot)
        # Rotate points
        sample.data.rotate_around_z(angle)
        # Rotate bboxes_3d
        if sample.bboxes_3d is not None:
            sample.bboxes_3d.rotate_around_z(angle)
        return sample


@manager.TRANSFORMS.add_component
class GlobalScale(TransformABC):
    """
    """

    def __init__(self, min_scale: float = 0.95, max_scale: float = 1.05):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalScale only supports lidar data!")
        factor = np.random.uniform(self.min_scale, self.max_scale)
        # Scale points
        sample.data.scale(factor)
        # Scale bboxes_3d
        if sample.bboxes_3d is not None:
            sample.bboxes_3d.scale(factor)
        return sample


@manager.TRANSFORMS.add_component
class GlobalTranslate(TransformABC):
    """
    Translate sample by a random offset.

    Args:
        translation_std (Union[float, List[float], Tuple[float]], optional):
            The standard deviation of the translation offset. Defaults to (.2, .2, .2).
    """

    def __init__(
            self,
            translation_std: Union[float, List[float], Tuple[float]] = (.2, .2,
                                                                        .2)):
        if not isinstance(translation_std, (list, tuple)):
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        self.translation_std = translation_std

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalScale only supports lidar data!")
        translation = np.random.normal(scale=self.translation_std, size=3)
        sample.data.translate(translation)
        sample.bboxes_3d.translate(translation)

        return sample


@manager.TRANSFORMS.add_component
class ShufflePoint(TransformABC):
    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("ShufflePoint only supports lidar data!")
        sample.data.shuffle()
        return sample


@manager.TRANSFORMS.add_component
class FilterBBoxOutsideRange(TransformABC):
    def __init__(self, point_cloud_range: Tuple[float]):
        self.point_cloud_range = np.asarray(point_cloud_range, dtype='float32')

    def __call__(self, sample: Sample):
        if sample.bboxes_3d.size == 0:
            return sample
        mask = sample.bboxes_3d.get_mask_of_bboxes_outside_range(
            self.point_cloud_range)
        sample.bboxes_3d = sample.bboxes_3d.masked_select(mask)
        sample.labels = sample.labels[mask]
        return sample


@manager.TRANSFORMS.add_component
class HardVoxelize(TransformABC):
    def __init__(self, point_cloud_range: Tuple[float],
                 voxel_size: Tuple[float], max_points_in_voxel: int,
                 max_voxel_num: int):
        self.max_points_in_voxel = max_points_in_voxel
        self.max_voxel_num = max_voxel_num
        self.voxel_size = np.asarray(voxel_size, dtype='float32')
        self.point_cloud_range = np.asarray(point_cloud_range, dtype='float32')
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) /
            self.voxel_size).astype('int32')

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("Voxelize only supports lidar data!")

        # Voxelize
        num_points, num_point_dim = sample.data.shape[0:2]
        voxels = np.zeros(
            (self.max_voxel_num, self.max_points_in_voxel, num_point_dim),
            dtype=sample.data.dtype)
        coords = np.zeros((self.max_voxel_num, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros((self.max_voxel_num, ), dtype=np.int32)
        grid_size_z, grid_size_y, grid_size_x = self.grid_size[::-1]
        grid_idx_to_voxel_idx = np.full((grid_size_z, grid_size_y, grid_size_x),
                                        -1,
                                        dtype=np.int32)

        num_voxels = points_to_voxel(
            sample.data, self.voxel_size, self.point_cloud_range,
            self.grid_size, voxels, coords, num_points_per_voxel,
            grid_idx_to_voxel_idx, self.max_points_in_voxel, self.max_voxel_num)

        voxels = voxels[:num_voxels]
        coords = coords[:num_voxels]
        num_points_per_voxel = num_points_per_voxel[:num_voxels]

        sample.voxels = voxels
        sample.coords = coords
        sample.num_points_per_voxel = num_points_per_voxel

        sample.pop('sweeps', None)
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

    def sampling(self, sample: Sample, num_samples_per_class: Dict[str, int]):
        existing_bboxes_3d = sample.bboxes_3d.copy()
        existing_velocities = None
        if sample.bboxes_3d.velocities is not None:
            existing_velocities = sample.bboxes_3d.velocities.copy()
        existing_labels = sample.labels.copy()
        existing_data = sample.data.copy()
        existing_difficulties = None if "difficulties" not in sample else sample.difficulties
        ignored_bboxes_3d = np.zeros(
            [0, existing_bboxes_3d.shape[1]], dtype=existing_bboxes_3d.dtype
        ) if "ignored_bboxes_3d" not in sample else sample.ignored_bboxes_3d
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
                        lidar_data[:, 0:3] += sampling_bboxes_3d[i, 0:3]
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


@manager.TRANSFORMS.add_component
class GenerateAnchors(TransformABC):
    """
    Generate SSD style anchors for PointPillars.

    Args:
        output_stride_factor (int): Output stride of the network.
        point_cloud_range (List[float]): [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size (List[float]): [x_size, y_size, z_size].
        anchor_configs (List[Dict[str, Any]]): Anchor configuration for each class. Attributes must include:
            "sizes": (List[float]) Anchor size (in wlh order).
            "strides": (List[float]) Anchor stride.
            "offsets": (List[float]) Anchor offset.
            "rotations": (List[float]): Anchor rotation.
            "matched_threshold": (float) IoU threshold for positive anchors.
            "unmatched_threshold": (float) IoU threshold for negative anchors.
        anchor_area_threshold (float): Threshold for filtering out anchor area. Defaults to 1.
    """

    def __init__(self,
                 output_stride_factor: int,
                 point_cloud_range: List[float],
                 voxel_size: List[float],
                 anchor_configs: List[Dict[str, Any]],
                 anchor_area_threshold: int = 1):
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.grid_size = np.round(
            (self.point_cloud_range[3:6] - self.point_cloud_range[:3]) /
            self.voxel_size).astype(np.int64)

        anchor_generators = [
            AnchorGeneratorStride(**anchor_cfg) for anchor_cfg in anchor_configs
        ]
        feature_map_size = self.grid_size[:2] // output_stride_factor
        feature_map_size = [*feature_map_size, 1][::-1]

        self._generate_anchors(feature_map_size, anchor_generators)
        self.anchor_area_threshold = anchor_area_threshold

    def _generate_anchors(self, feature_map_size, anchor_generators):
        anchors_list = []
        match_list = []
        unmatch_list = []
        for gen in anchor_generators:
            anchors = gen.generate(feature_map_size)
            anchors = anchors.reshape(
                [*anchors.shape[:3], -1, anchors.shape[-1]])
            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(
                np.full((num_anchors, ), gen.match_threshold, anchors.dtype))
            unmatch_list.append(
                np.full((num_anchors, ), gen.unmatch_threshold, anchors.dtype))

        anchors = np.concatenate(anchors_list, axis=-2)
        self.matched_thresholds = np.concatenate(match_list, axis=0)
        self.unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        self.anchors = anchors.reshape([-1, anchors.shape[-1]])
        self.anchors_bv = rbbox2d_to_near_bbox(self.anchors[:, [0, 1, 3, 4, 6]])

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GenerateAnchors only supports lidar data!")

        sample.anchors = self.anchors
        sample.matched_thresholds = self.matched_thresholds
        sample.unmatched_thresholds = self.unmatched_thresholds

        if self.anchor_area_threshold >= 0:
            # find anchors with area < threshold
            dense_voxel_map = F.sparse_sum_for_anchors_mask(
                sample.coords, tuple(self.grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = F.fused_get_anchors_area(
                dense_voxel_map, self.anchors_bv, self.voxel_size,
                self.point_cloud_range, self.grid_size)
            anchors_mask = anchors_area > self.anchor_area_threshold
            sample.anchors_mask = anchors_mask

        return sample


class AnchorGeneratorStride(object):
    def __init__(self,
                 sizes=[1.6, 3.9, 1.56],
                 anchor_strides=[0.4, 0.4, 1.0],
                 anchor_offsets=[0.2, -39.8, -1.78],
                 rotations=[0, np.pi / 2],
                 matched_threshold=-1,
                 unmatched_threshold=-1):
        self._sizes = sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._rotations = rotations
        self._match_threshold = matched_threshold
        self._unmatch_threshold = unmatched_threshold

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    def generate(self, feature_map_size):
        return F.create_anchors_3d_stride(feature_map_size, self._sizes,
                                          self._anchor_strides,
                                          self._anchor_offsets, self._rotations)


@manager.TRANSFORMS.add_component
class RandomObjectPerturb(TransformABC):
    """
    Randomly perturb (rotate and translate) each object.

    Args:
        rotation_range (Union[float, List[float], Tuple[float]], optional):
            Range of random rotation. Defaults to pi / 4.
        translation_std (Union[float, List[float], Tuple[float]], optional):
            Standard deviation of random translation. Defaults to 1.0.
        max_num_attempts (int): Maximum number of perturbation attempts. Defaults to 100.
    """

    def __init__(
            self,
            rotation_range: Union[float, List[float], Tuple[float]] = np.pi / 4,
            translation_std: Union[float, List[float], Tuple[float]] = 1.0,
            max_num_attempts: int = 100):
        if not isinstance(rotation_range, (list, tuple)):
            rotation_range = [-rotation_range, rotation_range]
        self.rotation_range = rotation_range
        if not isinstance(translation_std, (list, tuple)):
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        self.translation_std = translation_std
        self.max_num_attempts = max_num_attempts

    def __call__(self, sample: Sample):
        num_objects = sample.bboxes_3d.shape[0]
        rotation_noises = np.random.uniform(
            self.rotation_range[0],
            self.rotation_range[1],
            size=[num_objects, self.max_num_attempts])
        translation_noises = np.random.normal(
            scale=self.translation_std,
            size=[num_objects, self.max_num_attempts, 3])
        rotation_noises, translation_noises = F.noise_per_box(
            sample.bboxes_3d[:, [0, 1, 3, 4, 6]], sample.bboxes_3d.corners_2d,
            sample.ignored_bboxes_3d.corners_2d, rotation_noises,
            translation_noises)

        # perturb points w.r.t objects' centers (inplace operation)
        normals = F.corner_to_surface_normal(sample.bboxes_3d.corners_3d)
        point_masks = points_in_convex_polygon_3d_jit(sample.data[:, :3],
                                                      normals)
        F.perturb_object_points_(sample.data, sample.bboxes_3d[:, :3],
                                 point_masks, rotation_noises,
                                 translation_noises)

        # perturb bboxes_3d w.r.t to objects' centers (inplace operation)
        F.perturb_object_bboxes_3d_(sample.bboxes_3d, rotation_noises,
                                    translation_noises)

        return sample
