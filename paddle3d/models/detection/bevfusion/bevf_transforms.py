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

# This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/tree/main/mmdet3d/datasets/pipelines

import copy
import numbers

import cv2
import numpy as np
import paddle

from paddle3d.apis import manager
from paddle3d.sample import Sample
from paddle3d.transforms.base import TransformABC

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

__all__ = [
    'PointsRangeFilter', 'ResizeImage', 'NormalizeImage', 'PadImage',
    'PointShuffle'
]


@manager.TRANSFORMS.add_component
class PointsRangeFilter(TransformABC):
    """Filter points by the range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def in_range_3d(self, points, point_range):
        in_range_flags = ((points[:, 0] > point_range[0])
                          & (points[:, 1] > point_range[1])
                          & (points[:, 2] > point_range[2])
                          & (points[:, 0] < point_range[3])
                          & (points[:, 1] < point_range[4])
                          & (points[:, 2] < point_range[5]))
        return in_range_flags

    def __call__(self, sample):
        """Call function to filter points by the range.
        """

        points = sample['points']
        points_mask = self.in_range_3d(points, self.pcd_range)
        clean_points = points[points_mask]
        sample['points'] = clean_points

        return sample


@manager.TRANSFORMS.add_component
class ResizeImage(TransformABC):
    """Resize images & bbox & mask.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.
        """
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.
        """
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        """

        assert isinstance(img_scale, list) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(self.img_scale[0],
                                                        self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                if self.keep_ratio:
                    img, scale_factor = self.imrescale(
                        results[key][idx],
                        results['scale'],
                        interpolation='bilinear' if key == 'img' else 'nearest',
                        return_scale=True,
                        backend=self.backend)
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][idx].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    raise NotImplementedError
                results[key][idx] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def rescale_size(self, old_size, scale, return_scale=False):
        """Calculate the new size to be rescaled to.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, list):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
        else:
            raise TypeError(
                f'Scale must be a number or list of int, but got {type(scale)}')

        def _scale_size(size, scale):
            if isinstance(scale, (float, int)):
                scale = (scale, scale)
            w, h = size
            return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) +
                                                       0.5)

        new_size = _scale_size((w, h), scale_factor)

        if return_scale:
            return new_size, scale_factor
        else:
            return new_size

    def imrescale(self,
                  img,
                  scale,
                  return_scale=False,
                  interpolation='bilinear',
                  backend=None):
        """Resize image while keeping the aspect ratio.
        """
        h, w = img.shape[:2]
        new_size, scale_factor = self.rescale_size((w, h),
                                                   scale,
                                                   return_scale=True)
        rescaled_img = self.imresize(
            img, new_size, interpolation=interpolation, backend=backend)
        if return_scale:
            return rescaled_img, scale_factor
        else:
            return rescaled_img

    def imresize(self,
                 img,
                 size,
                 return_scale=False,
                 interpolation='bilinear',
                 out=None,
                 backend=None):
        """Resize image to a given size.
        """
        h, w = img.shape[:2]
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             f"Supported backends are 'cv2', 'pillow'")

        if backend == 'pillow':
            raise NotImplementedError
        else:
            resized_img = cv2.resize(
                img,
                size,
                dst=out,
                interpolation=cv2_interp_codes[interpolation])
        if not return_scale:
            return resized_img
        else:
            w_scale = size[0] / w
            h_scale = size[1] / h
            return resized_img, w_scale, h_scale

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        raise NotImplementedError

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        raise NotImplementedError

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        """
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'][0].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = list(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        return results


@manager.TRANSFORMS.add_component
class NormalizeImage(TransformABC):
    """Normalize the image.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def _imnormalize(self, img, mean, std, to_rgb=True):
        img = img.copy().astype(np.float32)
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def __call__(self, results):
        """Call function to normalize images.
        """
        for key in results.get('img_fields', ['img']):
            if key == 'img_depth':
                continue
            for idx in range(len(results['img'])):
                results[key][idx] = self._imnormalize(
                    results[key][idx], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@manager.TRANSFORMS.add_component
class PadImage(object):
    """Pad the image & mask.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def impad(self,
              img,
              *,
              shape=None,
              padding=None,
              pad_val=0,
              padding_mode='constant'):
        """Pad the given image to a certain shape or pad on all sides with
        specified padding mode and padding value.
        """

        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            padding = [0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0]]

        # check pad_val
        if isinstance(pad_val, list):
            assert len(pad_val) == img.shape[-1]
        elif not isinstance(pad_val, numbers.Number):
            raise TypeError('pad_val must be a int or a list. '
                            f'But received {type(pad_val)}')

        # check padding
        if isinstance(padding, list) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = [padding[0], padding[1], padding[0], padding[1]]
        elif isinstance(padding, numbers.Number):
            padding = [padding, padding, padding, padding]
        else:
            raise ValueError('Padding must be a int or a 2, or 4 element list.'
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

    def impad_to_multiple(self, img, divisor, pad_val=0):
        """Pad an image to ensure each edge to be multiple to some number.
        """
        pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
        pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
        return self.impad(img, shape=(pad_h, pad_w), pad_val=pad_val)

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = self.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                for idx in range(len(results[key])):
                    padded_img = self.impad_to_multiple(
                        results[key][idx],
                        self.size_divisor,
                        pad_val=self.pad_val)
                    results[key][idx] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        raise NotImplementedError

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        raise NotImplementedError

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        """
        self._pad_img(results)
        return results


@manager.TRANSFORMS.add_component
class SampleFilterByKey(TransformABC):
    """Collect data from the loader relevant to the specific task.
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                            'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pts_filename', 'transformation_3d_flow')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, sample):
        """Call function to filter sample by keys. The keys in ``meta_keys``
        """
        filtered_sample = Sample(path=sample.path, modality=sample.modality)
        filtered_sample.meta.id = sample.meta.id
        img_metas = {}

        for key in self.meta_keys:
            if key in sample:
                img_metas[key] = sample[key]

        filtered_sample['img_metas'] = img_metas
        for key in self.keys:
            filtered_sample[key] = sample[key]

        return filtered_sample


@manager.TRANSFORMS.add_component
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self, load_dim=6, use_dim=[0, 1, 2], shift_height=False):
        self.shift_height = shift_height
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

        self.load_dim = load_dim
        self.use_dim = use_dim

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (np.ndarray): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]

        if self.shift_height:
            raise NotImplementedError

        results['points'] = points

        return results


@manager.TRANSFORMS.add_component
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False,
                 point_cloud_angle_range=None):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

        if point_cloud_angle_range is not None:
            self.filter_by_angle = True
            self.point_cloud_angle_range = point_cloud_angle_range
            print(point_cloud_angle_range)
        else:
            self.filter_by_angle = False
            # self.point_cloud_angle_range = point_cloud_angle_range

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def filter_point_by_angle(self, points):
        if isinstance(points, np.ndarray):
            points_numpy = points
        else:
            raise NotImplementedError
        pts_phi = (np.arctan(points_numpy[:, 0] / points_numpy[:, 1]) +
                   (points_numpy[:, 1] < 0) * np.pi + np.pi * 2) % (np.pi * 2)

        pts_phi[pts_phi > np.pi] -= np.pi * 2
        pts_phi = pts_phi / np.pi * 180

        assert np.all(-180 <= pts_phi) and np.all(pts_phi <= 180)

        filt = np.logical_and(pts_phi >= self.point_cloud_angle_range[0],
                              pts_phi <= self.point_cloud_angle_range[1])
        return points[filt]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray): Multi-sweep point cloud arrays.
        """
        points = results['points']
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                # points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        if self.filter_by_angle:
            points = self.filter_point_by_angle(points)

        points = points[:, self.use_dim]
        results['points'] = points
        return results


@manager.TRANSFORMS.add_component
class GlobalRotScaleTrans(TransformABC):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of ranslation
            noise. This apply random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if not isinstance(self.translation_std, (list, tuple, np.ndarray)):
            translation_std = [
                self.translation_std, self.translation_std, self.translation_std
            ]
        else:
            translation_std = self.translation_std
        translation_std = np.array(translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'][:, :3] += trans_factor
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key][:, :3] += trans_factor

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        rotation = self.rot_range
        if not isinstance(rotation, list):
            rotation = [-rotation, rotation]
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        rot_sin = np.sin(noise_rotation)
        rot_cos = np.cos(noise_rotation)
        rot_mat_T = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                              [0, 0, 1]])

        # rotate bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key]) != 0:
                gt_bboxes_3d = input_dict['gt_bboxes_3d']
                gt_bboxes_3d[:, :3] = gt_bboxes_3d[:, :3] @ rot_mat_T
                gt_bboxes_3d[:, 6] += noise_rotation

                if gt_bboxes_3d.shape[1] == 9:
                    # rotate velo vector
                    gt_bboxes_3d[:, 7:
                                 9] = gt_bboxes_3d[:, 7:9] @ rot_mat_T[:2, :2]
                input_dict['gt_bboxes_3d'] = gt_bboxes_3d
                input_dict['pcd_rotation'] = rot_mat_T

        # rotate points in clock-wise
        rot_sin = np.sin(-noise_rotation)
        rot_cos = np.cos(-noise_rotation)
        rot_mat_T = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                              [0, 0, 1]])
        input_dict['points'][:, :3] = input_dict['points'][:, :3] @ rot_mat_T.T

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']

        # scale points
        input_dict['points'][:, :3] *= scale

        if self.shift_height:
            raise NotImplementedError

        # scale bboxes
        for key in input_dict['bbox3d_fields']:
            input_dict[key][:, :6] *= scale
            input_dict[key][:, 7:] *= scale

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])

        return input_dict


@manager.TRANSFORMS.add_component
class RandomFlip3D(TransformABC):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(RandomFlip3D, self).__init__()
        self.sync_2d = sync_2d
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](np.array(
                [], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1

        # flip bboxes
        for key in input_dict['bbox3d_fields']:
            if direction == 'horizontal':
                input_dict[key][:, 1::7] = -input_dict[key][:, 1::7]
                input_dict[key][:, 6] = -input_dict[key][:, 6] + np.pi
            else:
                raise NotImplementedError

        # flip points
        if direction == 'horizontal':
            input_dict['points'][:, 1] = -input_dict['points'][:, 1]
        else:
            raise NotImplementedError

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        if 'pcd_horizontal_flip' not in input_dict:
            flip_horizontal = True if np.random.rand(
            ) < self.flip_ratio_bev_horizontal else False
            input_dict['pcd_horizontal_flip'] = flip_horizontal
        if 'pcd_vertical_flip' not in input_dict:
            flip_vertical = True if np.random.rand(
            ) < self.flip_ratio_bev_vertical else False
            input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])

        return input_dict


@manager.TRANSFORMS.add_component
class PointShuffle(TransformABC):
    def __call__(self, input_dict):
        input_dict['points'] = input_dict['points'][np.random.permutation(
            input_dict['points'].shape[0])]
        return input_dict
