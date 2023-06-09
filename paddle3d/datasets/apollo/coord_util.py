# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------
# Modified from BEV-LaneDet (https://github.com/gigo-team/bev_lane_det)
# ------------------------------------------------------------------------

import numpy as np
import cv2


def transform_matrix(
        translation,
        rotation,
        inverse: bool = False,
) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)
    if inverse:
        rot_inv = rotation.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def image2ego_byheight(image_points,
                       camera_intrinsic,
                       camera2ego_matrix,
                       height=0):
    """
    :param image_points: 3*n [[u,v,1],,,,] for example
    :param camera_intsic:
    :param camera2ego_matrix:
    :param height:
    :return: ego_points
    """
    ego2camera_matrix = np.linalg.inv(camera2ego_matrix)
    camera_intrinsic_inv = np.linalg.inv(camera_intrinsic)
    R_inv = ego2camera_matrix[:3, :3].T
    T = ego2camera_matrix[:3, 3]
    mat1 = np.dot(np.dot(R_inv, camera_intrinsic_inv), image_points)
    mat2 = np.dot(R_inv, T)
    Zc = (height + mat2[2]) / mat1[2]
    points_ego = Zc * mat1 - np.expand_dims(mat2, 1)
    return points_ego


def ego2image(ego_points, camera_intrinsic, camera2ego_matrix):
    """
    :param ego_points:  3*n
    :param camera_intrinsic: 3*3
    :param camera2ego_matrix:  4*4
    :return:
    """
    ego2camera_matrix = np.linalg.inv(camera2ego_matrix)
    camera_points = np.dot(ego2camera_matrix[:3, :3], ego_points) + \
                    ego2camera_matrix[:3, 3].reshape(3, 1)
    image_points_ = camera_intrinsic @ camera_points
    image_points = image_points_ / image_points_[2]
    return image_points


def image_undistort(img, camera_intrinsic, distort_params, mode='pinhole'):
    h, w = img.shape[:2]
    if mode == 'pinhole':
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_intrinsic, distort_params, None, camera_intrinsic, (w, h), 5)
    elif mode == 'fisheye':
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
            camera_intrinsic, distort_params, None, camera_intrinsic, (w, h), 5)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


def IPM2ego_matrix(ipm_center=None,
                   m_per_pixel=None,
                   ipm_points=None,
                   ego_points=None):
    if ipm_points is None:
        center_x, center_y = ipm_center[0] * m_per_pixel, ipm_center[
            1] * m_per_pixel
        M = np.array([[-m_per_pixel, 0, center_x], [0, -m_per_pixel, center_y]])
    else:
        pts1 = np.float32(ipm_points)
        pts2 = np.float32(ego_points)
        M = cv2.getAffineTransform(pts1, pts2)
    return M
