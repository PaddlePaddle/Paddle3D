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
from pathlib import Path
from typing import List, Union
import cv2
import paddle
import numpy as np
from PIL import Image
from einops import rearrange

from paddle3d.apis import manager
from paddle3d.datasets.kitti import kitti_utils
from paddle3d.datasets.semantic_kitti.semantic_kitti import \
    SemanticKITTIDataset
from paddle3d.geometries import PointCloud
from paddle3d.geometries.bbox import points_in_convex_polygon_3d_jit
from paddle3d.models.detection.bevfusion.utils import generate_guassian_depth_target, map_pointcloud_to_image
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
from paddle3d.utils.logger import logger

__all__ = [
    "LoadImage", "LoadPointCloud", "RemoveCameraInvisiblePointsKITTI",
    "RemoveCameraInvisiblePointsKITTIV2", "LoadSemanticKITTIRange"
]


@manager.TRANSFORMS.add_component
class LoadImage(TransformABC):
    """
    """
    _READER_MAPPER = {"cv2": cv2.imread, "pillow": Image.open}

    def __init__(self,
                 to_chw: bool = True,
                 to_rgb: bool = True,
                 reader: str = "cv2"):
        if reader not in self._READER_MAPPER.keys():
            raise ValueError('Unsupported reader {}'.format(reader))

        self.reader = reader
        self.to_rgb = to_rgb
        self.to_chw = to_chw

    def __call__(self, sample: Sample) -> Sample:
        """
        """
        sample.data = np.array(self._READER_MAPPER[self.reader](sample.path))

        sample.meta.image_reader = self.reader
        sample.meta.image_format = "bgr" if self.reader == "cv2" else "rgb"
        sample.meta.channel_order = "hwc"

        if sample.meta.image_format != "rgb" and self.to_rgb:
            if sample.meta.image_format == "bgr":
                sample.data = cv2.cvtColor(sample.data, cv2.COLOR_BGR2RGB)
                sample.meta.image_format = "rgb"
            else:
                raise RuntimeError('Unsupported image format {}'.format(
                    sample.meta.image_format))
        elif sample.meta.image_format != "bgr" and (self.to_rgb is False):
            if sample.meta.image_format == "rgb":
                sample.data = sample.data[:, :, ::-1]
                sample.meta.image_format = "bgr"
            else:
                raise RuntimeError('Unsupported image format {}'.format(
                    sample.meta.image_format))

        if self.to_chw:
            sample.data = sample.data.transpose((2, 0, 1))
            sample.meta.channel_order = "chw"

        return sample


@manager.TRANSFORMS.add_component
class LoadPointCloud(TransformABC):
    """
    Load point cloud.

    Args:
        dim: The dimension of each point.
        use_dim: The dimension of each point to use.
        use_time_lag: Whether to use time lag.
        sweep_remove_radius: The radius within which points are removed in sweeps.
    """

    def __init__(self,
                 dim,
                 use_dim: Union[int, List[int]] = None,
                 use_time_lag: bool = False,
                 sweep_remove_radius: float = 1):
        self.dim = dim
        self.use_dim = range(use_dim) if isinstance(use_dim, int) else use_dim
        self.use_time_lag = use_time_lag
        self.sweep_remove_radius = sweep_remove_radius

    def __call__(self, sample: Sample):
        """
        """
        if sample.modality != "lidar":
            raise ValueError('{} Only Support samples in modality lidar'.format(
                self.__class__.__name__))

        if sample.data is not None:
            raise ValueError(
                'The data for this sample has been processed before.')

        data = np.fromfile(sample.path, np.float32).reshape(-1, self.dim)

        if self.use_dim is not None:
            data = data[:, self.use_dim]

        if self.use_time_lag:
            time_lag = np.zeros((data.shape[0], 1), dtype=data.dtype)
            data = np.hstack([data, time_lag])

        if len(sample.sweeps) > 0:
            data_sweep_list = [
                data,
            ]
            for i in np.random.choice(
                    len(sample.sweeps), len(sample.sweeps), replace=False):
                sweep = sample.sweeps[i]
                sweep_data = np.fromfile(sweep.path, np.float32).reshape(
                    -1, self.dim)
                if self.use_dim:
                    sweep_data = sweep_data[:, self.use_dim]
                sweep_data = sweep_data.T

                # Remove points that are in a certain radius from origin.
                x_filter_mask = np.abs(
                    sweep_data[0, :]) < self.sweep_remove_radius
                y_filter_mask = np.abs(
                    sweep_data[1, :]) < self.sweep_remove_radius
                not_close = np.logical_not(
                    np.logical_and(x_filter_mask, y_filter_mask))
                sweep_data = sweep_data[:, not_close]

                # Homogeneous transform of current sample to reference coordinate
                if sweep.meta.ref_from_curr is not None:
                    sweep_data[:3, :] = sweep.meta.ref_from_curr.dot(
                        np.vstack((sweep_data[:3, :],
                                   np.ones(sweep_data.shape[1]))))[:3, :]
                sweep_data = sweep_data.T
                if self.use_time_lag:
                    curr_time_lag = sweep.meta.time_lag * np.ones(
                        (sweep_data.shape[0], 1)).astype(sweep_data.dtype)
                    sweep_data = np.hstack([sweep_data, curr_time_lag])
                data_sweep_list.append(sweep_data)
            data = np.concatenate(data_sweep_list, axis=0)

        sample.data = PointCloud(data)
        return sample


@manager.TRANSFORMS.add_component
class RemoveCameraInvisiblePointsKITTI(TransformABC):
    """
    Remove camera invisible points for KITTI dataset.
    """

    def __call__(self, sample: Sample):
        calibs = sample.calibs
        C, Rinv, T = kitti_utils.projection_matrix_decomposition(calibs[2])

        im_path = (Path(sample.path).parents[1] / "image_2" / Path(
            sample.path).stem).with_suffix(".png")

        if os.path.exists(im_path):
            im_shape = cv2.imread(str(im_path)).shape[:2]
        else:
            im_shape = (375, 1242)
        im_shape = np.array(im_shape, dtype=np.int32)

        im_bbox = [0, 0, im_shape[1], im_shape[0]]
        frustum = F.get_frustum(im_bbox, C)
        frustum = (Rinv @ (frustum - T).T).T
        frustum = kitti_utils.coord_camera_to_velodyne(frustum, calibs)
        frustum_normals = F.corner_to_surface_normal(frustum[None, ...])

        indices = points_in_convex_polygon_3d_jit(sample.data[:, :3],
                                                  frustum_normals)
        sample.data = sample.data[indices.reshape([-1])]

        return sample


@manager.TRANSFORMS.add_component
class RemoveCameraInvisiblePointsKITTIV2(TransformABC):
    """
    Remove camera invisible points for KITTI dataset, unlike `RemoveCameraInvisiblePointsKITTI` which projects image plane to a frustum,
    this version projects poinst into image plane and remove the points outside the image boundary.
    """

    def __init__(self):
        self.V2C = None
        self.R0 = None

    def __call__(self, sample: Sample):
        calibs = sample.calibs
        self.R0 = calibs[4]
        self.V2C = calibs[5]
        self.P2 = calibs[2]

        im_path = (Path(sample.path).parents[1] / "image_2" / Path(
            sample.path).stem).with_suffix(".png")

        if os.path.exists(im_path):
            im_shape = cv2.imread(str(im_path)).shape[:2]
        else:
            im_shape = (375, 1242)
        im_shape = np.array(im_shape, dtype=np.int32)

        pts = sample.data[:, 0:3]
        # lidar to rect
        pts_lidar_hom = self.cart_to_hom(pts)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))

        # rect to img
        pts_img, pts_rect_depth = self.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0,
                                    pts_img[:, 0] < im_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0,
                                    pts_img[:, 1] < im_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        sample.data = sample.data[pts_valid_flag]
        return sample

    def cart_to_hom(self, pts):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_img(self, pts_rect):
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[
            3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth


@manager.TRANSFORMS.add_component
class LoadSemanticKITTIRange(TransformABC):
    """
    Load SemanticKITTI range image.
    Please refer to <https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/laserscan.py>.

    Args:
        project_label (bool, optional): Whether project label to range view or not.
    """

    def __init__(self, project_label=True):
        self.project_label = project_label
        self.proj_H = 64
        self.proj_W = 1024
        self.upper_inclination = 3. / 180. * np.pi
        self.lower_inclination = -25. / 180. * np.pi
        self.fov = self.upper_inclination - self.lower_inclination

        self.remap_lut = SemanticKITTIDataset.build_remap_lut()

    def _remap_semantic_labels(self, sem_label):
        """
        Remap semantic labels to cross entropy format.
        Please refer to <https://github.com/PRBonn/semantic-kitti-api/blob/master/remap_semantic_labels.py>.
        """

        return self.remap_lut[sem_label]

    def __call__(self, sample: Sample) -> Sample:
        raw_scan = np.fromfile(sample.path, dtype=np.float32).reshape((-1, 4))
        points = raw_scan[:, 0:3]
        remissions = raw_scan[:, 3]

        # get depth of all points (L-2 norm of [x, y, z])
        depth = np.linalg.norm(points, ord=2, axis=1)

        # get angles of all points
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (
            pitch + abs(self.lower_inclination)) / self.fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        proj_x_copy = np.copy(
            proj_x
        )  # save a copy in original order, for each point, where it is in the range image

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        proj_y_copy = np.copy(
            proj_y
        )  # save a copy in original order, for each point, where it is in the range image

        # unproj_range_copy = np.copy(depth)   # copy of depth in original order

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = points[order]
        remission = remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # projected range image - [H,W] range (-1 is no data)
        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        proj_remission = np.full((self.proj_H, self.proj_W),
                                 -1,
                                 dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points
        proj_remission[proj_y, proj_x] = remission
        proj_idx[proj_y, proj_x] = indices
        proj_mask = proj_idx > 0  # mask containing for each pixel, if it contains a point or not

        sample.data = np.concatenate([
            proj_range[None, ...],
            proj_xyz.transpose([2, 0, 1]), proj_remission[None, ...]
        ])

        sample.meta["proj_mask"] = proj_mask.astype(np.float32)
        sample.meta["proj_x"] = proj_x_copy
        sample.meta["proj_y"] = proj_y_copy

        if sample.labels is not None:
            # load labels
            raw_label = np.fromfile(
                sample.labels, dtype=np.uint32).reshape((-1))
            # only fill in attribute if the right size
            if raw_label.shape[0] == points.shape[0]:
                sem_label = raw_label & 0xFFFF  # semantic label in lower half
                sem_label = self._remap_semantic_labels(sem_label)
                # inst_label = raw_label >> 16  # instance id in upper half
            else:
                logger.error("Point cloud shape: {}".format(points.shape))
                logger.error("Label shape: {}".format(raw_label.shape))
                raise ValueError(
                    "Scan and Label don't contain same number of points. {}".
                    format(sample.path))
            # # sanity check
            # assert ((sem_label + (inst_label << 16) == raw_label).all())

            if self.project_label:
                # project label to range view
                # semantics
                proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                          dtype=np.int32)  # [H,W]  label
                proj_sem_label[proj_mask] = sem_label[proj_idx[proj_mask]]

                # # instances
                # proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                #                            dtype=np.int32)  # [H,W]  label
                # proj_inst_label[proj_mask] = self.inst_label[proj_idx[proj_mask]]

                sample.labels = proj_sem_label.astype(np.int64)[None, ...]
            else:
                sample.labels = sem_label.astype(np.int64)

        return sample


@manager.TRANSFORMS.add_component
class LoadSemanticKITTIPointCloud(TransformABC):
    """
    Load SemanticKITTI range image.
    Please refer to <https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/laserscan.py>.
    """

    def __init__(self, use_dim: List[int] = None):
        self.proj_H = 64
        self.proj_W = 1024
        self.upper_inclination = 3. / 180. * np.pi
        self.lower_inclination = -25. / 180. * np.pi
        self.fov = self.upper_inclination - self.lower_inclination

        self.remap_lut = SemanticKITTIDataset.build_remap_lut()

        self.use_dim = use_dim

    def _remap_semantic_labels(self, sem_label):
        """
        Remap semantic labels to cross entropy format.
        Please refer to <https://github.com/PRBonn/semantic-kitti-api/blob/master/remap_semantic_labels.py>.
        """

        return self.remap_lut[sem_label]

    def __call__(self, sample: Sample) -> Sample:
        raw_scan = np.fromfile(sample.path, dtype=np.float32).reshape(-1, 4)
        points = raw_scan[:, 0:3]

        sample.data = PointCloud(raw_scan[:, self.use_dim])

        if sample.labels is not None:
            # load labels
            raw_label = np.fromfile(sample.labels, dtype=np.int32).reshape(-1)
            # only fill in attribute if the right size
            if raw_label.shape[0] == points.shape[0]:
                sem_label = raw_label & 0xFFFF  # semantic label in lower half
                sem_label = self._remap_semantic_labels(sem_label)
                # self.inst_label = raw_label >> 16  # instance id in upper half
            else:
                logger.error("Point cloud shape: {}".format(points.shape))
                logger.error("Label shape: {}".format(raw_label.shape))
                raise ValueError(
                    "Scan and Label don't contain same number of points. {}".
                    format(sample.path))
            # # sanity check
            # assert ((sem_label + (inst_label << 16) == raw_label).all())

            sample.labels = sem_label

        return sample


@manager.TRANSFORMS.add_component
class LoadMultiViewImageFromFiles(TransformABC):
    """
    load multi-view image from files

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Default: False.
        color_type (str): Color type of the file. Default: -1.
            - -1: cv2.IMREAD_UNCHANGED
            -  0: cv2.IMREAD_GRAYSCALE
            -  1: cv2.IMREAD_COLOR
    """

    def __init__(self,
                 to_float32=False,
                 project_pts_to_img_depth=False,
                 cam_depth_range=[4.0, 45.0, 1.0],
                 constant_std=0.5,
                 imread_flag=-1):
        self.to_float32 = to_float32
        self.project_pts_to_img_depth = project_pts_to_img_depth
        self.cam_depth_range = cam_depth_range
        self.constant_std = constant_std
        self.imread_flag = imread_flag

    def __call__(self, sample):
        """
        Call function to load multi-view image from files.
        """
        filename = sample['img_filename']

        img = np.stack(
            [cv2.imread(name, self.imread_flag) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        sample['filename'] = filename

        sample['img'] = [img[..., i] for i in range(img.shape[-1])]
        sample['img_shape'] = img.shape
        sample['ori_shape'] = img.shape

        sample['pad_shape'] = img.shape
        # sample['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]

        sample['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        sample['img_fields'] = ['img']

        if self.project_pts_to_img_depth:
            sample['img_depth'] = []
            for i in range(len(sample['img'])):
                depth = map_pointcloud_to_image(
                    sample['points'],
                    sample['img'][i],
                    sample['caminfo'][i]['sensor2lidar_rotation'],
                    sample['caminfo'][i]['sensor2lidar_translation'],
                    sample['caminfo'][i]['cam_intrinsic'],
                    show=False)
                guassian_depth, min_depth, std_var = generate_guassian_depth_target(
                    paddle.to_tensor(depth).unsqueeze(0),
                    stride=8,
                    cam_depth_range=self.cam_depth_range,
                    constant_std=self.constant_std)
                depth = paddle.concat(
                    [min_depth[0].unsqueeze(-1), guassian_depth[0]], axis=-1)
                sample['img_depth'].append(depth)
        return sample


@manager.TRANSFORMS.add_component
class LoadAnnotations3D(TransformABC):
    """
    load annotation
    """

    def __init__(
            self,
            with_bbox_3d=True,
            with_label_3d=True,
            with_attr_label=False,
            with_mask_3d=False,
            with_seg_3d=False,
    ):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d

    def _load_bboxes_3d(self, sample) -> Sample:
        """
        """
        sample['gt_bboxes_3d'] = sample['ann_info']['gt_bboxes_3d']
        sample['bbox3d_fields'].append('gt_bboxes_3d')
        return sample

    def _load_labels_3d(self, sample) -> Sample:
        """
        """
        sample['gt_labels_3d'] = sample['ann_info']['gt_labels_3d']
        return sample

    def _load_attr_labels(self, sample) -> Sample:
        """
        """
        sample['attr_labels'] = sample['ann_info']['attr_labels']
        return sample

    def __call__(self, sample) -> Sample:
        """Call function to load multiple types annotations.
        """
        if self.with_bbox_3d:
            sample = self._load_bboxes_3d(sample)
            if sample is None:
                return None

        if self.with_label_3d:
            sample = self._load_labels_3d(sample)

        if self.with_attr_label:
            sample = self._load_attr_labels(sample)

        return sample


@manager.TRANSFORMS.add_component
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            sweeps_num=5,
            to_float32=False,
            pad_empty_sweeps=False,
            sweep_range=[3, 27],
            sweeps_id=None,
            imread_flag=-1,  #'unchanged'
            sensors=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            test_mode=True,
            prob=1.0,
    ):

        self.sweeps_num = sweeps_num
        self.to_float32 = to_float32
        self.imread_flag = imread_flag
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, sample):
        """Call function to load multi-view sweep image from filenames.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = sample['img']
        img_timestamp = sample['img_timestamp']
        lidar_timestamp = sample['timestamp']
        img_timestamp = [
            lidar_timestamp - timestamp for timestamp in img_timestamp
        ]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(sample['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (
                    self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend(
                    [time + mean_time for time in img_timestamp])
                for j in range(nums):
                    sample['filename'].append(sample['filename'][j])
                    sample['lidar2img'].append(np.copy(sample['lidar2img'][j]))
                    sample['intrinsics'].append(
                        np.copy(sample['intrinsics'][j]))
                    sample['extrinsics'].append(
                        np.copy(sample['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(sample['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(sample['sweeps']))
            elif self.test_mode:
                choices = [
                    int((self.sweep_range[0] + self.sweep_range[1]) / 2) - 1
                ]
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(sample['sweeps']):
                        sweep_range = list(
                            range(
                                self.sweep_range[0],
                                min(self.sweep_range[1],
                                    len(sample['sweeps']))))
                    else:
                        sweep_range = list(
                            range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(
                        sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [
                        int((self.sweep_range[0] + self.sweep_range[1]) / 2) - 1
                    ]

            for idx in choices:
                sweep_idx = min(idx, len(sample['sweeps']) - 1)
                sweep = sample['sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = sample['sweeps'][sweep_idx - 1]
                sample['filename'].extend(
                    [sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([
                    cv2.imread(sweep[sensor]['data_path'], self.imread_flag)
                    for sensor in self.sensors
                ],
                               axis=-1)

                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [
                    lidar_timestamp - sweep[sensor]['timestamp'] / 1e6
                    for sensor in self.sensors
                ]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    sample['lidar2img'].append(sweep[sensor]['lidar2img'])
                    sample['intrinsics'].append(sweep[sensor]['intrinsics'])
                    sample['extrinsics'].append(sweep[sensor]['extrinsics'])
        sample['img'] = sweep_imgs_list
        sample['timestamp'] = timestamp_imgs_list

        return sample


@manager.TRANSFORMS.add_component
class LoadMapsFromFiles(object):
    def __init__(self, k=None):
        self.k = k

    def __call__(self, sample) -> Sample:
        map_filename = sample['map_filename']
        maps = np.load(map_filename)
        map_mask = maps['arr_0'].astype(np.float32)

        maps = map_mask.transpose((2, 0, 1))
        sample['gt_map'] = maps
        maps = rearrange(
            maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps = maps.reshape(256, 3 * 256)
        sample['map_shape'] = maps.shape
        sample['maps'] = maps
        return sample
