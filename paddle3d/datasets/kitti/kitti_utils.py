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
from typing import List, Tuple

import numpy as np
import paddle

from paddle3d.geometries import BBoxes2D, BBoxes3D, CoordMode
from paddle3d.sample import Sample


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


def filter_fake_result(detection: Sample):
    if detection.get('bboxes_3d', None) is None:
        return

    box3d = detection.bboxes_3d
    scores = detection.confidences
    labels = detection.labels
    box_list = []
    score_list = []
    label_list = []
    for i in range(box3d.shape[0]):
        if scores[i] < 0:
            continue
        box_list.append(box3d[i])
        score_list.append(scores[i])
        label_list.append(labels[i])
    if len(box_list) == 0:
        detection.bboxes_3d = None
        detection.labels = None
        detection.confidences = None
    else:
        detection.bboxes_3d = BBoxes3D(
            np.asarray(box_list),
            origin=box3d.origin,
            rot_axis=box3d.rot_axis,
            coordmode=box3d.coordmode)
        detection.labels = np.asarray(label_list)
        detection.confidences = np.asarray(score_list)


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/object3d_kitti.py#L18
    """

    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(
            label[2]
        )  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(
            label[6]), float(label[7])),
                              dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array(
            (float(label[11]), float(label[12]), float(label[13])),
            dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)], [0, 1, 0],
                      [-np.sin(self.ry), 0,
                       np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
            % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
               self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str


class Calibration(object):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/calibration_kitti.py#L23
    """

    def __init__(self, calib_dict):
        calib = calib_dict

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1),
                                              dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4),
                                             dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4),
                                                dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(
            np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[
            3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)),
            axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))),
                                       axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :,
                                                            1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1),
                                x2.reshape(-1, 1), y2.reshape(-1, 1)),
                               axis=1)
        boxes_corner = np.concatenate(
            (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    # GUPNET relative function
    def alpha2ry(self, alpha, u):
        """
        Get rotation_y by alpha + theta - 180
        alpha : Observation angle of object, ranging [-pi..pi]
        x : Object center x to the camera center (x-W/2), in pixels
        rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """
        ry = alpha + np.arctan2(u - self.cu, self.fu)

        if ry > np.pi:
            ry -= 2 * np.pi
        if ry < -np.pi:
            ry += 2 * np.pi

        return ry

    # GUPNET relative function
    def ry2alpha(self, ry, u):
        alpha = ry - np.arctan2(u - self.cu, self.fu)

        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi

        return alpha

    # GUPNET relative function
    def flip(self, img_size):
        wsize = 4
        hsize = 2
        p2ds = (np.concatenate([
            np.expand_dims(
                np.tile(
                    np.expand_dims(np.linspace(0, img_size[0], wsize), 0),
                    [hsize, 1]), -1),
            np.expand_dims(
                np.tile(
                    np.expand_dims(np.linspace(0, img_size[1], hsize), 1),
                    [1, wsize]), -1),
            np.linspace(2, 78, wsize * hsize).reshape(hsize, wsize, 1)
        ], -1)).reshape(-1, 3)
        p3ds = self.img_to_rect(p2ds[:, 0:1], p2ds[:, 1:2], p2ds[:, 2:3])
        p3ds[:, 0] *= -1
        p2ds[:, 0] = img_size[0] - p2ds[:, 0]

        # self.P2[0,3] *= -1
        cos_matrix = np.zeros([wsize * hsize, 2, 7])
        cos_matrix[:, 0, 0] = p3ds[:, 0]
        cos_matrix[:, 0, 1] = cos_matrix[:, 1, 2] = p3ds[:, 2]
        cos_matrix[:, 1, 0] = p3ds[:, 1]
        cos_matrix[:, 0, 3] = cos_matrix[:, 1, 4] = 1
        cos_matrix[:, :, -2] = -p2ds[:, :2]
        cos_matrix[:, :, -1] = (-p2ds[:, :2] * p3ds[:, 2:3])
        new_calib = np.linalg.svd(cos_matrix.reshape(-1, 7))[-1][-1]
        new_calib /= new_calib[-1]

        new_calib_matrix = np.zeros([4, 3]).astype(np.float32)
        new_calib_matrix[0, 0] = new_calib_matrix[1, 1] = new_calib[0]
        new_calib_matrix[2, 0:2] = new_calib[1:3]
        new_calib_matrix[3, :] = new_calib[3:6]
        new_calib_matrix[-1, -1] = self.P2[-1, -1]
        self.P2 = new_calib_matrix.T
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)
