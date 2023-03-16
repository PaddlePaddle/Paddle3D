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

import numbers
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
from paddle3d.apis import manager
from paddle3d.geometries.bbox import BBoxes3D, CoordMode, points_in_convex_polygon_3d_jit
from paddle3d.ops import voxelize
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
from paddle3d.transforms.functional import points_to_voxel
from paddle3d.utils import box_utils

__all__ = [
    "RandomHorizontalFlip", "RandomVerticalFlip", "GlobalRotate", "GlobalScale",
    "GlobalTranslate", "ShufflePoint", "SamplePoint", "SamplePointByVoxels",
    "FilterPointOutsideRange", "FilterBBoxOutsideRange", "HardVoxelize",
    "RandomObjectPerturb", "ConvertBoxFormat", "ResizeShortestEdge",
    "RandomContrast", "RandomBrightness", "RandomSaturation",
    "ToVisionBasedBox", "PhotoMetricDistortionMultiViewImage",
    "RandomScaleImageMultiViewImage"
]


@manager.TRANSFORMS.add_component
class RandomHorizontalFlip(TransformABC):
    """
    Note:
        If the inputs are pixel indices, they are flipped by `(W - 1 - x, H - 1 - y)`.
        If the inputs are floating point coordinates, they are flipped by `(W - x, H - y)`.
    """

    def __init__(self, prob: float = 0.5, input_type='pixel_indices'):
        self.prob = prob
        self.input_type = input_type

    def __call__(self, sample: Sample):
        if np.random.random() < self.prob:
            if sample.modality == "image":
                sample.data = F.horizontal_flip(sample.data)
                h, w, c = sample.data.shape
            elif sample.modality == "lidar":
                sample.data.flip(axis=1)

            if self.input_type == 'pixel_indices':
                # Flip camera intrinsics
                if "camera_intrinsic" in sample.meta:
                    sample.meta.camera_intrinsic[
                        0, 2] = w - sample.meta.camera_intrinsic[0, 2] - 1

                # Flip bbox
                if sample.bboxes_3d is not None:
                    sample.bboxes_3d.horizontal_flip()
                if sample.bboxes_2d is not None and sample.modality == "image":
                    sample.bboxes_2d.horizontal_flip(image_width=w)

            elif self.input_type == 'floating_point_coordinates':
                # Flip camera intrinsics
                if "camera_intrinsic" in sample.meta:
                    sample.meta.camera_intrinsic[
                        0, 2] = w - sample.meta.camera_intrinsic[0, 2]

                # Flip bbox
                if sample.bboxes_3d is not None:
                    sample.bboxes_3d.horizontal_flip_coords()
                if sample.bboxes_2d is not None and sample.modality == "image":
                    sample.bboxes_2d.horizontal_flip_coords(image_width=w)
        return sample


@manager.TRANSFORMS.add_component
class ToVisionBasedBox(TransformABC):
    """
    """

    def __call__(self, sample: Sample):
        bboxes_3d_new = sample.bboxes_3d.to_vision_based_3d_box()
        sample.bboxes_3d = BBoxes3D(
            bboxes_3d_new,
            origin=[.5, 1, .5],
            coordmode=CoordMode.KittiCamera,
            rot_axis=1)
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

    def __init__(self,
                 min_scale: float = 0.95,
                 max_scale: float = 1.05,
                 size=None):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.size = size

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalScale only supports lidar data!")
        factor = np.random.uniform(
            self.min_scale, self.max_scale, size=self.size)
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
        distribution (str):
            The random distribution. Defaults to normal.
    """

    def __init__(
            self,
            translation_std: Union[float, List[float], Tuple[float]] = (.2, .2,
                                                                        .2),
            distribution="normal"):
        if not isinstance(translation_std, (list, tuple)):
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        self.translation_std = translation_std
        self.distribution = distribution

    def __call__(self, sample: Sample):
        if sample.modality != "lidar":
            raise ValueError("GlobalScale only supports lidar data!")
        if self.distribution not in ["normal", "uniform"]:
            raise ValueError(
                "GlobalScale only supports normal and uniform random distribution!"
            )

        if self.distribution == "normal":
            translation = np.random.normal(scale=self.translation_std, size=3)
        elif self.distribution == "uniform":
            translation = np.random.uniform(
                low=-self.translation_std[0],
                high=self.translation_std[0],
                size=3)
        else:
            raise ValueError(
                "GlobalScale only supports normal and uniform random distribution!"
            )

        sample.data.translate(translation)
        if sample.bboxes_3d is not None:
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
class ConvertBoxFormat(TransformABC):
    def __call__(self, sample: Sample):
        # convert boxes from [x,y,z,w,l,h,yaw] to [x,y,z,l,w,h,heading], bottom_center -> obj_center
        bboxes_3d = box_utils.boxes3d_kitti_lidar_to_lidar(sample.bboxes_3d)

        # limit heading
        bboxes_3d[:, -1] = box_utils.limit_period(
            bboxes_3d[:, -1], offset=0.5, period=2 * np.pi)

        # stack labels into gt_boxes, label starts from 1, instead of 0.
        labels = sample.labels + 1
        bboxes_3d = np.concatenate(
            [bboxes_3d, labels.reshape(-1, 1).astype(np.float32)], axis=-1)
        sample.bboxes_3d = bboxes_3d
        sample.pop('labels', None)

        return sample


@manager.TRANSFORMS.add_component
class SamplePoint(TransformABC):
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, sample: Sample):
        sample = F.sample_point(sample, self.num_points)

        return sample


@manager.TRANSFORMS.add_component
class SamplePointByVoxels(TransformABC):
    def __init__(self, voxel_size, max_points_per_voxel, max_num_of_voxels,
                 num_points, point_cloud_range):
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.max_num_of_voxels = max_num_of_voxels
        self.num_points = num_points
        self.point_cloud_range = point_cloud_range

    def transform_points_to_voxels(self, sample):
        points = sample.data
        points = paddle.to_tensor(points)
        voxels, coordinates, num_points, voxels_num = voxelize.hard_voxelize(
            points, self.voxel_size, self.point_cloud_range,
            self.max_points_per_voxel, self.max_num_of_voxels)
        voxels = voxels[:voxels_num, :, :].numpy()
        coordinates = coordinates[:voxels_num, :].numpy()
        num_points = num_points[:voxels_num, :].numpy()

        sample['voxels'] = voxels
        sample['voxel_coords'] = coordinates
        sample['voxel_num_points'] = num_points

        return sample

    def sample_points_by_voxels(self, sample):
        if self.num_points == -1:  # dynamic voxelization !
            return sample

        # voxelization
        sample = self.transform_points_to_voxels(sample)

        points = sample['voxels'][:, 0]  # remain only one point per voxel

        sample.data = points
        # sampling
        sample = F.sample_point(sample, self.num_points)
        sample.pop('voxels')
        sample.pop('voxel_coords')
        sample.pop('voxel_num_points')

        return sample

    def __call__(self, sample):
        return self.sample_points_by_voxels(sample)


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
class FilterPointOutsideRange(TransformABC):
    def __init__(self, point_cloud_range: Tuple[float]):
        self.point_cloud_range = np.asarray(point_cloud_range, dtype='float32')

    def __call__(self, sample: Sample):
        mask = sample.data.get_mask_of_points_outside_range(
            self.point_cloud_range)
        sample.data = sample.data[mask]
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


@manager.TRANSFORMS.add_component
class ResizeShortestEdge(TransformABC):
    """
    """

    def __init__(self,
                 short_edge_length,
                 max_size,
                 sample_style="range",
                 interp=Image.BILINEAR):
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.interp = interp
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!")

    def __call__(self, sample: Sample):
        h, w = sample.data.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0],
                                     self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        newh, neww = self.get_output_shape(h, w, size, self.max_size)
        sample.data = self.apply_image(sample.data, h, w, newh, neww)
        sample.image_sizes = np.asarray((h, w))
        if "camera_intrinsic" in sample.meta:
            sample.meta.camera_intrinsic = self.apply_intrinsics(
                sample.meta.camera_intrinsic, h, w, newh, neww)
        if sample.bboxes_2d is not None and sample.modality == "image":
            sample.bboxes_2d.resize(h, w, newh, neww)
        return sample

    def apply_image(self, img, h, w, newh, neww):
        assert len(img.shape) <= 4

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((neww, newh), self.interp)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = paddle.to_tensor(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.reshape(shape_4d).transpose([2, 3, 0, 1])  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[self.interp]
            align_corners = None if mode == "nearest" else False
            img = nn.functional.interpolate(
                img, (newh, neww), mode=mode, align_corners=align_corners)
            shape[:2] = (newh, neww)
            ret = img.transpose([2, 3, 0,
                                 1]).reshape(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_intrinsics(self, intrinsics, h, w, newh, neww):
        assert intrinsics.shape == (3, 3)
        assert intrinsics[0, 1] == 0  # undistorted
        assert np.allclose(intrinsics,
                           np.triu(intrinsics))  # check if upper triangular

        factor_x = neww / w
        factor_y = newh / h
        new_intrinsics = intrinsics * np.float32([factor_x, factor_y, 1
                                                  ]).reshape(3, 1)
        return new_intrinsics

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int,
                         max_size: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


@manager.TRANSFORMS.add_component
class RandomContrast(TransformABC):
    """
    Randomly transforms image contrast.
    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast
    """

    def __init__(self, intensity_min: float, intensity_max: float):
        super().__init__()
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __call__(self, sample: Sample):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        sample.data = F.blend_transform(
            sample.data,
            src_image=sample.data.mean(),
            src_weight=1 - w,
            dst_weight=w)
        return sample


@manager.TRANSFORMS.add_component
class RandomBrightness(TransformABC):
    """
    Randomly transforms image contrast.
    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast
    """

    def __init__(self, intensity_min: float, intensity_max: float):
        super().__init__()
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __call__(self, sample: Sample):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        sample.data = F.blend_transform(
            sample.data, src_image=0, src_weight=1 - w, dst_weight=w)
        return sample


@manager.TRANSFORMS.add_component
class RandomSaturation(TransformABC):
    """
    Randomly transforms image contrast.
    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast
    """

    def __init__(self, intensity_min: float, intensity_max: float):
        super().__init__()
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __call__(self, sample: Sample):
        assert sample.data.shape[
            -1] == 3, "RandomSaturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = sample.data.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        sample.data = F.blend_transform(
            sample.data, src_image=grayscale, src_weight=1 - w, dst_weight=w)
        return sample


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (paddle.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        paddle.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - np.floor(val / period + offset) * period


@manager.TRANSFORMS.add_component
class SampleRangeFilter(object):
    """
    Filter samples by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def in_range_bev(self, box_range, gt_bboxes_3d):
        """
        Check whether the boxes are in the given range.
        """
        in_range_flags = ((gt_bboxes_3d[:, 0] > box_range[0])
                          & (gt_bboxes_3d[:, 1] > box_range[1])
                          & (gt_bboxes_3d[:, 0] < box_range[2])
                          & (gt_bboxes_3d[:, 1] < box_range[3]))
        return in_range_flags

    def limit_yaw(self, gt_bboxes_3d, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw.
            period (float): The expected period.
        """
        gt_bboxes_3d[:, 6] = limit_period(gt_bboxes_3d[:, 6], offset, period)
        return gt_bboxes_3d

    def __call__(self, sample):
        """Call function to filter objects by the range.

        Args:
            sample (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the Sample.
        """
        if isinstance(sample['gt_bboxes_3d'], (BBoxes3D, np.ndarray)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        else:
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = sample['gt_bboxes_3d']
        gt_labels_3d = sample['gt_labels_3d']

        mask = self.in_range_bev(bev_range, gt_bboxes_3d)
        gt_bboxes_3d = gt_bboxes_3d[mask]

        gt_labels_3d = gt_labels_3d[mask.astype(np.bool_)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d = self.limit_yaw(
            gt_bboxes_3d, offset=0.5, period=2 * np.pi)
        sample['gt_bboxes_3d'] = gt_bboxes_3d
        sample['gt_labels_3d'] = gt_labels_3d

        return sample


@manager.TRANSFORMS.add_component
class SampleNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, sample):
        """Call function to filter objects by their names.

        Args:
            sample (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the Sample.
        """
        gt_labels_3d = sample['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        sample['gt_bboxes_3d'] = sample['gt_bboxes_3d'][gt_bboxes_mask]
        sample['gt_labels_3d'] = sample['gt_labels_3d'][gt_bboxes_mask]

        return sample


@manager.TRANSFORMS.add_component
class ResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, sample_aug_cfg=None, training=True):
        self.sample_aug_cfg = sample_aug_cfg

        self.training = training

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        imgs = sample["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            sample['intrinsics'][
                i][:3, :3] = ida_mat @ sample['intrinsics'][i][:3, :3]

        sample["img"] = new_imgs
        sample['lidar2img'] = [
            sample['intrinsics'][i] @ sample['extrinsics'][i].T
            for i in range(len(sample['extrinsics']))
        ]

        return sample

    def _get_rot(self, h):

        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = np.eye(2)
        ida_tran = np.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= np.array(crop[:2])
        if flip:
            A = np.array([[-1, 0], [0, 1]])
            b = np.array([crop[2] - crop[0], 0])

            ida_rot = np.matmul(A, ida_rot)
            ida_tran = np.matmul(A, ida_tran) + b

        A = self._get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = np.matmul(A, -b) + b
        ida_rot = np.matmul(A, ida_rot)
        ida_tran = np.matmul(A, ida_tran) + b
        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.sample_aug_cfg["H"], self.sample_aug_cfg["W"]
        fH, fW = self.sample_aug_cfg["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.sample_aug_cfg["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.sample_aug_cfg["bot_pct_lim"])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.sample_aug_cfg["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.sample_aug_cfg["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.sample_aug_cfg["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@manager.TRANSFORMS.add_component
class MSResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self,
                 sample_aug_cfg=None,
                 training=True,
                 view_num=1,
                 center_size=2.0):
        self.sample_aug_cfg = sample_aug_cfg
        self.training = training
        self.view_num = view_num
        self.center_size = center_size

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            sample (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = sample["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        copy_intrinsics = []
        copy_extrinsics = []
        for i in range(self.view_num):
            copy_intrinsics.append(np.copy(sample['intrinsics'][i]))
            copy_extrinsics.append(np.copy(sample['extrinsics'][i]))

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            sample['intrinsics'][
                i][:3, :3] = ida_mat @ sample['intrinsics'][i][:3, :3]

        resize, resize_dims, crop, flip, rotate = self._crop_augmentation(
            resize)
        for i in range(self.view_num):
            img = Image.fromarray(np.copy(np.uint8(imgs[i])))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            copy_intrinsics[i][:3, :3] = ida_mat @ copy_intrinsics[i][:3, :3]
            sample['intrinsics'].append(copy_intrinsics[i])
            sample['extrinsics'].append(copy_extrinsics[i])
            sample['filename'].append(sample['filename'][i].replace(
                ".jpg", "_crop.jpg"))
            sample['timestamp'].append(sample['timestamp'][i])

        sample["img"] = new_imgs
        sample['lidar2img'] = [
            sample['intrinsics'][i] @ sample['extrinsics'][i].T
            for i in range(len(sample['extrinsics']))
        ]
        return sample

    def _get_rot(self, h):

        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = np.eye(2)
        ida_tran = np.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= np.array(crop[:2])
        if flip:
            A = np.array([[-1, 0], [0, 1]])
            b = np.array([crop[2] - crop[0], 0])

            ida_rot = np.matmul(A, ida_rot)
            ida_tran = np.matmul(A, ida_tran) + b

        A = self._get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = np.matmul(A, -b) + b
        ida_rot = np.matmul(A, ida_rot)
        ida_tran = np.matmul(A, ida_tran) + b
        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.sample_aug_cfg["H"], self.sample_aug_cfg["W"]
        fH, fW = self.sample_aug_cfg["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.sample_aug_cfg["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.sample_aug_cfg["bot_pct_lim"])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.sample_aug_cfg["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.sample_aug_cfg["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.sample_aug_cfg["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _crop_augmentation(self, resize):
        H, W = self.sample_aug_cfg["H"], self.sample_aug_cfg["W"]
        fH, fW = self.sample_aug_cfg["final_dim"]
        resize = self.center_size * resize
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int(max(0, newH - fH) / 2)
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate


@manager.TRANSFORMS.add_component
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
            self,
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0],
            reverse_angle=False,
            training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)

        self.rotate_bev_along_z(results, rot_angle)

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        # results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_bev_along_z(self, results, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.array(
                results["lidar2img"][view]).astype('float32') @ rot_mat_inv
            results["extrinsics"][view] = np.array(
                results["extrinsics"][view]).astype('float32') @ rot_mat_inv

        if self.reverse_angle:
            rot_angle = np.array(-1 * angle)
        else:
            rot_angle = np.array(angle)

        rot_cos = np.cos(rot_angle)
        rot_sin = np.sin(rot_angle)

        rot_mat = np.array([[
            rot_cos,
            -rot_sin,
            0,
        ], [
            rot_sin,
            rot_cos,
            0,
        ], [0, 0, 1]])
        results.gt_bboxes_3d[:, :3] = results.gt_bboxes_3d[:, :3] @ rot_mat
        results.gt_bboxes_3d[:, 6] += rot_angle
        results.gt_bboxes_3d[:, 7:
                             9] = results.gt_bboxes_3d[:, 7:9] @ rot_mat[:2, :2]

    def scale_xyz(self, results, scale_ratio):
        rot_mat = np.array([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])

        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.array(
                results["lidar2img"][view]).astype('float32') @ rot_mat_inv
            results["extrinsics"][view] = np.array(
                rot_mat_inv.T @ results["extrinsics"][view]).astype('float32')

        results.gt_bboxes_3d[:, :6] *= scale_ratio
        results.gt_bboxes_3d[:, 7:] *= scale_ratio
        return


@manager.TRANSFORMS.add_component
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, sample):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        sample['img'] = [
            F.normalize_use_cv2(img, self.mean, self.std, self.to_rgb)
            for img in sample['img']
        ]
        sample['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return sample


def impad(img, *, shape=None, padding=None, pad_val=0, padding_mode='constant'):
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img


def impad_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)


@manager.TRANSFORMS.add_component
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if self.size is not None:
            padded_img = [
                impad(img, shape=self.size, pad_val=self.pad_val)
                for img in sample['img']
            ]
        elif self.size_divisor is not None:
            padded_img = [
                impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in sample['img']
            ]
        sample['img_shape'] = [img.shape for img in sample['img']]
        sample['img'] = padded_img
        sample['pad_shape'] = [img.shape for img in padded_img]
        sample['pad_fixed_size'] = self.size
        sample['pad_size_divisor'] = self.size_divisor
        return sample


@manager.TRANSFORMS.add_component
class SampleFilerByKey(object):
    """Collect data from the loader relevant to the specific task.
    """

    def __init__(
            self,
            keys,
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                       'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                       'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                       'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                       'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                       'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                       'transformation_3d_flow', 'scene_token', 'can_bus')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, sample):
        """Call function to filter sample by keys. The keys in ``meta_keys``

        Args:
            sample (dict): Result dict contains the data.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        filtered_sample = Sample(path=sample.path, modality=sample.modality)
        filtered_sample.meta.id = sample.meta.id

        for key in self.meta_keys:
            if key in sample:
                filtered_sample.meta[key] = sample[key]

        for key in self.keys:
            filtered_sample[key] = sample[key]
        return filtered_sample


@manager.TRANSFORMS.add_component
class PhotoMetricDistortionMultiViewImage(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of np.random.contrast is in
    second or second to last.
    1. np.random.brightness
    2. np.random.contrast (mode 0)
    3. convert color from BGR to HSV
    4. np.random.saturation
    5. np.random.hue
    6. convert color from HSV to BGR
    7. np.random.contrast (mode 1)
    8. np.random.y swap channels

    This class is modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L99

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert_color_factory(self, src, dst):

        code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

        def convert_color(img):
            out_img = cv2.cvtColor(img, code)
            return out_img

        convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
            image.

        Args:
            img (ndarray or str): The input image.

        Returns:
            ndarray: The converted {dst.upper()} image.
        """

        return convert_color

    def __call__(self, sample):
        """Call function to perform photometric distortion on images.
        Args:
            sample (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = sample['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # np.random.brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta,
                                          self.brightness_delta)
                img += delta

            # mode == 0 --> do np.random.contrast first
            # mode == 1 --> do np.random.contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = self.convert_color_factory('bgr', 'hsv')(img)

            # np.random.saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(self.saturation_lower,
                                                 self.saturation_upper)

            # np.random.hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta,
                                                 self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = self.convert_color_factory('hsv', 'bgr')(img)

            # np.random.contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            # np.random.y swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        sample['img'] = new_imgs
        return sample


@manager.TRANSFORMS.add_component
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    This class is modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L289
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales) == 1

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            sample (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in sample['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in sample['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        sample['img'] = [
            imresize(img, (x_size[idx], y_size[idx]), return_scale=False)
            for idx, img in enumerate(sample['img'])
        ]
        lidar2img = [scale_factor @ l2i for l2i in sample['lidar2img']]
        sample['lidar2img'] = lidar2img
        sample['img_shape'] = [img.shape for img in sample['img']]
        sample['ori_shape'] = [img.shape for img in sample['img']]

        return sample


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None,
             backend=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``cv2`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = 'cv2'
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

if Image is not None:
    pillow_interp_codes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'box': Image.BOX,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }
