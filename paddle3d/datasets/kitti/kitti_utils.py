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

from typing import List, Tuple

import numpy as np

from paddle3d.geometries import BBoxes2D, BBoxes3D, CoordMode


# kitti record fields
# type, truncated, occluded, alpha, xmin, ymin, xmax, ymax, dh, dw, dl, lx, ly, lz, ry
def camera_record_to_object(
        kitti_records: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    if kitti_records.shape[0] == 0:
        bboxes_2d = BBoxes2D(np.zeros([0, 4]))
        bboxes_3d = BBoxes3D(
            np.zeros([0, 7]),
            origin=[.5, 1, .5],
            coordmode=CoordMode.KittiCamera,
            rot_axis=1)
        labels = []
    else:
        centers = kitti_records[:, 11:14]
        dims = kitti_records[:, 8:11]
        yaws = kitti_records[:, 14:15]
        bboxes_3d = BBoxes3D(
            np.concatenate([centers, dims, yaws], axis=1),
            origin=[.5, 1, .5],
            coordmode=CoordMode.KittiCamera,
            rot_axis=1)
        bboxes_2d = BBoxes2D(kitti_records[:, 4:8])
        labels = kitti_records[:, 0]

    return bboxes_2d, bboxes_3d, labels


# lidar record fields
# type, truncated, occluded, alpha, xmin, ymin, xmax, ymax, dw, dl, dh, lx, ly, lz, rz
def lidar_record_to_object(
        kitti_records: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    if kitti_records.shape[0] == 0:
        bboxes_2d = BBoxes2D(np.zeros([0, 4]))
        bboxes_3d = BBoxes3D(
            np.zeros([0, 7]),
            origin=[0.5, 0.5, 0.],
            coordmode=CoordMode.KittiLidar,
            rot_axis=2)
        cls_names = []
    else:
        centers = kitti_records[:, 11:14]
        dims = kitti_records[:, 8:11]
        yaws = kitti_records[:, 14:15]
        bboxes_3d = BBoxes3D(
            np.concatenate([centers, dims, yaws], axis=1),
            origin=[0.5, 0.5, 0.],
            coordmode=CoordMode.KittiLidar,
            rot_axis=2)
        bboxes_2d = BBoxes2D(kitti_records[:, 4:8])
        cls_names = kitti_records[:, 0]

    return bboxes_2d, bboxes_3d, cls_names


def project_camera_to_velodyne(kitti_records: np.ndarray,
                               calibration_info: Tuple[np.ndarray]):
    """
    """
    if kitti_records.shape[0] == 0:
        return kitti_records
    # locations
    kitti_records[:, 11:14] = coord_camera_to_velodyne(kitti_records[:, 11:14],
                                                       calibration_info)

    # rotations
    # In kitti records, dimensions order is hwl format, but standard camera order is lhw format.
    # We exchange lhw format to wlh format, which equal to yaw = yaw + np.pi / 2
    kitti_records[:, 8:11] = kitti_records[:, [9, 10, 8]]

    return kitti_records


def box_lidar_to_camera(bboxes_3d: BBoxes3D,
                        calibration_info: Tuple[np.ndarray]):
    ox, oy, oz = bboxes_3d.origin
    xyz_lidar = bboxes_3d[..., 0:3]
    w, l, h = bboxes_3d[..., 3:4], bboxes_3d[..., 4:5], bboxes_3d[..., 5:6]
    r = bboxes_3d[..., 6:7]
    xyz = coord_velodyne_to_camera(xyz_lidar, calibration_info)
    cam_bboxes = BBoxes3D(
        data=np.concatenate([xyz, l, h, w, r], axis=-1),
        coordmode=CoordMode.KittiCamera,
        velocities=None,
        origin=[1 - oy, 1 - oz, ox],
        rot_axis=1)
    return cam_bboxes


def coord_camera_to_velodyne(points: np.ndarray,
                             calibration_info: Tuple[np.ndarray]):
    """
    """
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calibration_info[4]

    V2C = np.eye(4)
    V2C[:3, :4] = calibration_info[5]

    pads = np.ones([points.shape[0], 1])
    points = np.concatenate([points, pads], axis=1)

    points = points @ np.linalg.inv(R0_rect @ V2C).T
    points = points[:, :3]

    return points


def coord_velodyne_to_camera(points: np.ndarray, calibration_info: np.ndarray):
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calibration_info[4]

    V2C = np.eye(4)
    V2C[:3, :4] = calibration_info[5]

    pads = np.ones([points.shape[0], 1])
    points = np.concatenate([points, pads], axis=1)

    points = points @ (R0_rect @ V2C).T
    points = points[:, :3]

    return points


def project_velodyne_to_camera(pointcloud: np.ndarray,
                               calibration_info: np.ndarray, image_shape):
    """
    """
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calibration_info[4]
    V2C = np.eye(4)
    V2C[:3, :4] = calibration_info[5]
    P2 = np.eye(4)
    P2[:3, :4] = calibration_info[2]

    intensity = pointcloud[:, 3:4]
    pointcloud = pointcloud[:, :3]

    pads = np.ones([pointcloud.shape[0], 1])
    pointcloud = np.concatenate([pointcloud, pads], axis=1).T
    cam = P2 @ R0_rect @ V2C @ pointcloud

    h, w = image_shape
    valid_indexes = cam[2, :] >= 0
    cam = cam[:2, :] / cam[2, :]

    # keep only the points in front of the camera
    valid_indexes &= cam[0, :] > 0
    valid_indexes &= cam[0, :] < w
    valid_indexes &= cam[1, :] > 0
    valid_indexes &= cam[1, :] < h

    pointcloud = pointcloud.T[valid_indexes, :3]
    intensity = intensity[valid_indexes, :]
    pointcloud = np.concatenate([pointcloud, intensity], axis=1)

    return pointcloud


def assess_object_difficulties(kitti_records: np.ndarray,
                               min_height_thresh: List = [40, 25, 25],
                               max_occlusion_thresh: List = [0, 1, 2],
                               max_truncation_thresh: List = [0.15, 0.3, 0.5]):
    num_objects = kitti_records.shape[0]
    if num_objects == 0:
        return np.full((num_objects, ), -1, dtype=np.int32)
    heights = kitti_records[:, 7] - kitti_records[:, 5]  # bboxes_2d heights
    occlusions = kitti_records[:, 2]
    truncations = kitti_records[:, 1]

    easy_mask = np.ones((num_objects, ), dtype=bool)
    moderate_mask = np.ones((num_objects, ), dtype=bool)
    hard_mask = np.ones((num_objects, ), dtype=bool)

    easy_mask[np.where((heights <= min_height_thresh[0])
                       | (occlusions > max_occlusion_thresh[0])
                       | (truncations > max_truncation_thresh[0]))] = False
    moderate_mask[np.where((heights <= min_height_thresh[1])
                           | (occlusions > max_occlusion_thresh[1])
                           | (truncations > max_truncation_thresh[1]))] = False
    hard_mask[np.where((heights <= min_height_thresh[2])
                       | (occlusions > max_occlusion_thresh[2])
                       | (truncations > max_truncation_thresh[2]))] = False

    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    difficulties = np.full((num_objects, ), -1, dtype=np.int32)
    difficulties[is_hard] = 2
    difficulties[is_moderate] = 1
    difficulties[is_easy] = 0

    return difficulties


def projection_matrix_decomposition(proj):
    """
    Calculate the camera calibration matrix, the invert of 3x3 rotation matrix,
    and the 3x1 translation vector that projects 3D points into the camera.
    Where:
        proj = C @ [R|T]

    Please refer to:
        <https://github.com/traveller59/second.pytorch/blob/master/second/core/box_np_ops.py#L507>
    """

    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    T = Cinv @ CT

    return C, Rinv, T
