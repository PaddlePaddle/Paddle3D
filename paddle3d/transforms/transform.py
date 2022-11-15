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

from typing import Any, List, Tuple, Union

import numpy as np
from PIL import Image

import paddle.nn as nn
from paddle3d.apis import manager
from paddle3d.geometries.bbox import BBoxes3D, CoordMode, points_in_convex_polygon_3d_jit
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
from paddle3d.transforms.functional import points_to_voxel

__all__ = [
    "RandomHorizontalFlip", "RandomVerticalFlip", "GlobalRotate", "GlobalScale",
    "GlobalTranslate", "ShufflePoint", "FilterBBoxOutsideRange", "HardVoxelize",
    "RandomObjectPerturb", "ResizeShortestEdge", "RandomContrast", "RandomBrightness",
    "RandomSaturation", "ToVisionBasedBox"
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
    def __init__(self, short_edge_length, max_size, sample_style="range", interp=Image.BILINEAR):
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
                f" Got {short_edge_length}!"
            )
    
    def __call__(self, sample: Sample):
        h, w = sample.data.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        newh, neww = self.get_output_shape(h, w, size, self.max_size)
        sample.data = self.apply_image(sample.data, h, w, newh, neww)
        sample.image_sizes = np.asarray((h, w)) 
        if "camera_intrinsic" in sample.meta:
            sample.meta.camera_intrinsic = self.apply_intrinsics(sample.meta.camera_intrinsic, h, w, newh, neww)
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
                img, (newh, neww), mode=mode, align_corners=align_corners
            )
            shape[:2] = (newh, neww)
            ret = img.transpose([2, 3, 0, 1]).reshape(shape).numpy()  # nchw -> hw(c)

        return ret
    
    def apply_intrinsics(self, intrinsics, h, w, newh, neww):
        assert intrinsics.shape == (3, 3)
        assert intrinsics[0, 1] == 0  # undistorted
        assert np.allclose(intrinsics, np.triu(intrinsics))  # check if upper triangular

        factor_x = neww / w
        factor_y = newh / h
        new_intrinsics = intrinsics * np.float32([factor_x, factor_y, 1]).reshape(3, 1)
        return new_intrinsics

    @staticmethod
    def get_output_shape(
        oldh: int, oldw: int, short_edge_length: int, max_size: int) -> Tuple[int, int]:
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
        sample.data = F.blend_transform(sample.data, src_image=sample.data.mean(), src_weight=1 - w, dst_weight=w)
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
        sample.data = F.blend_transform(sample.data, src_image=0, src_weight=1 - w, dst_weight=w)
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
        assert sample.data.shape[-1] == 3, "RandomSaturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = sample.data.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        sample.data = F.blend_transform(sample.data, src_image=grayscale, src_weight=1 - w, dst_weight=w)
        return sample
