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
"""
The geometry transformations for bounding box is modified from
https://github.com/tianweiy/CenterPoint and https://github.com/traveller59/second.pytorch

Ths copyright of tianweiy/CenterPoint is as follows:
MIT License [see LICENSE for details].

Ths copyright of traveller59/second is as follows:
MIT License [see LICENSE for details].
"""

from enum import Enum
from typing import List

import numba
import numpy as np
import scipy
from scipy.spatial import Delaunay
from pyquaternion import Quaternion

from paddle3d.geometries.structure import _Structure


class CoordMode(Enum):
    """
    """
    #                      z front
    #                     /
    #                    /
    #                   0 ------> x right
    #                   |
    #                   |
    #                   v
    #                 y down
    KittiCamera = 0

    #                  up z
    #                   ^   x front
    #                   |  /
    #                   | /
    #    left y <------ 0
    KittiLidar = 1

    #                  up z
    #                   ^   y front
    #                   |  /
    #                   | /
    #                   0 ------> x right
    NuScenesLidar = 2


class BBoxes2D(_Structure):
    """
    """

    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 2:
            raise ValueError('Illegal 2D box data with number of dim {}'.format(
                data.ndim))

        if data.shape[1] != 4:
            raise ValueError('Illegal 2D box data with shape {}'.format(
                data.shape))

    def scale(self, factor: float):
        ...

    def translate(self, translation: np.ndarray):
        ...

    def rotate(self, rotation: np.ndarray):
        ...

    def horizontal_flip(self, image_width: float):
        """
        The inputs are pixel indices, they are flipped by `(W - 1 - x, H - 1 - y)`.
        """
        self[:, 0] = image_width - self[:, 0] - 1

    def horizontal_flip_coords(self, image_width: float):
        """
        The inputs are floating point coordinates, they are flipped by `(W - x, H - y)`.
        """
        self[:, 0], self[:,
                         2] = image_width - self[:, 2], image_width - self[:, 0]

    def vertical_flip(self, image_height: float):
        self[:, 1] = image_height - self[:, 1] - 1

    def resize(self, h: int, w: int, newh: int, neww: int):
        factor_x = neww / w
        factor_y = newh / h
        self[:, 0::2] *= factor_x
        self[:, 1::2] *= factor_y


class BBoxes3D(_Structure):
    """
    """

    def __init__(self,
                 data: np.ndarray,
                 coordmode: CoordMode = 0,
                 velocities: List[float] = None,
                 origin: List[float] = [0.5, 0.5, 0.5],
                 rot_axis: int = 2):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.coordmode = coordmode
        self.velocities = velocities
        self.origin = origin
        self.rot_axis = rot_axis

    @property
    def corners_3d(self):
        # corners_3d format: x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0
        dx, dy, dz = self[:, 3:6].T
        b = dz.shape[0]

        x_corners = np.array([[0., 0., 0., 0., 1., 1., 1., 1.]],
                             self.dtype).repeat(
                                 b, axis=0)
        y_corners = np.array([[0., 0., 1., 1., 0., 0., 1., 1.]],
                             self.dtype).repeat(
                                 b, axis=0)
        z_corners = np.array([[0., 1., 1., 0., 0., 1., 1., 0.]],
                             self.dtype).repeat(
                                 b, axis=0)

        x_corners = (
            dx[:, np.newaxis] * (x_corners - self.origin[0]))[:, :, np.newaxis]
        y_corners = (
            dy[:, np.newaxis] * (y_corners - self.origin[1]))[:, :, np.newaxis]
        z_corners = (
            dz[:, np.newaxis] * (z_corners - self.origin[2]))[:, :, np.newaxis]
        corners = np.concatenate([x_corners, y_corners, z_corners], axis=-1)

        angle = self[:, -1]
        corners = rotation_3d_in_axis(corners, angle, axis=self.rot_axis)
        centers = self[:, 0:3][:, np.newaxis, :]
        corners += centers

        return corners

    @property
    def corners_2d(self):
        # corners_2d format: x0y0, x0y1, x1y1, x1y0
        dx, dy = self[:, 3:5].T
        b = dy.shape[0]

        x_corners = np.array([[0., 0., 1., 1.]], self.dtype).repeat(b, axis=0)
        y_corners = np.array([[0., 1., 1., 0.]], self.dtype).repeat(b, axis=0)

        x_corners = (
            dx[:, np.newaxis] * (x_corners - self.origin[0]))[:, :, np.newaxis]
        y_corners = (
            dy[:, np.newaxis] * (y_corners - self.origin[1]))[:, :, np.newaxis]
        corners = np.concatenate([x_corners, y_corners], axis=-1)

        angle = self[:, -1]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        rotation_matrix = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]],
                                   dtype=self.dtype)
        #rotation_matrix = rotation_matrix.transpose([2, 0, 1])
        #corners = corners @ rotation_matrix #TODO(luoqianhui)
        corners = np.einsum("aij,jka->aik", corners, rotation_matrix)

        centers = self[:, 0:2][:, np.newaxis, :]
        corners += centers

        return corners

    def scale(self, factor: float):
        """
        """
        # Scale x, y, z, w, l, h, except the orientation
        self[..., :-1] = self[..., :-1] * factor

        # Scale velocities
        if self.velocities is not None:
            self.velocities[..., :] = self.velocities[..., :] * factor

    def translate(self, translation: np.ndarray):
        self[..., :3] = self[..., :3] + translation

    def rotate_around_z(self, angle: np.ndarray):
        # Rotation matrix around the z-axis
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        rotation_matrix = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=self.dtype)

        # Rotate x,y,z
        self[..., :3] = self[..., :3] @ rotation_matrix

        # Rotate velocities
        if self.velocities is not None:
            self.velocities[..., :2] = (np.hstack([
                self.velocities[..., :2],
                np.zeros(
                    (self.velocities.shape[0], 1), dtype=self.velocities.dtype)
            ]) @ rotation_matrix)[..., :2]

        # Update orientation
        self[..., -1] += angle

    def horizontal_flip(self):
        """
        The inputs are pixel indices
        """
        self[:, 0] = -self[:, 0]
        if self.velocities is not None:
            self.velocities[:, 0] = -self.velocities[:, 0]
        self[:,
             -1] = -self[:,
                         -1] + 2 * np.pi  # TODO(luoqianhui): CHECK THIS 2 * np.pi is needed

    def horizontal_flip_coords(self):
        """
        The inputs are floating point coordinates
        """
        new_box3d_quat = np.stack(
            [self[:, 3], -self[:, 2], -self[:, 1], self[:, 0]], 1)
        self[:, :4] = new_box3d_quat
        self[:, 4] = -self[:, 4]

    def to_vision_based_3d_box(self):
        height, width, length = self[:, 3:4], self[:, 4:5], self[:, 5:6]
        x, y, z = self[:, 0:1], self[:, 1:2], self[:, 2:3]
        rotation = self[:, 6]
        tvec = np.concatenate([x, y - height / 2, z], axis=1)
        box_pose = []
        for i in range(rotation.shape[0]):
            wxyz = Quaternion(
                Quaternion(axis=[1, 0, 0], radians=np.pi / 2) * Quaternion(
                    axis=[0, 0, 1], radians=-rotation[i]))
            box_pose.append(wxyz.elements.astype(np.float32))
        box_pose = np.stack(box_pose, axis=0)
        box3d_new = np.concatenate([box_pose, tvec, width, length, height],
                                   axis=1)
        return box3d_new

    def vertical_flip(self):
        self[:, 1] = -self[:, 1]
        if self.velocities is not None:
            self.velocities[:, 1] = -self.velocities[:, 1]
        self[:, -1] = -self[:, -1] + np.pi

    @staticmethod
    def limit_period(val, offset: float = 0.5, period: float = np.pi):
        return val - np.floor(val / period + offset) * period

    def get_mask_of_bboxes_outside_range(self, point_cloud_range: np.ndarray):
        bboxes_bev = self.corners_2d
        # Represent the bev range as a bounding box
        limit_polygons = minmax_range_3d_to_corner_2d(point_cloud_range)
        mask = points_in_convex_polygon_2d(
            bboxes_bev.reshape(-1, 2), limit_polygons)
        return np.any(mask.reshape(-1, 4), axis=1)

    def masked_select(self, mask):
        selected_data = self[mask]
        selected_velocities = self.velocities
        if self.velocities is not None:
            selected_velocities = self.velocities[mask]
        selected_bbox = BBoxes3D(selected_data, self.coordmode,
                                 selected_velocities, self.origin,
                                 self.rot_axis)
        return selected_bbox


@numba.jit(nopython=True)
def get_mask_points_in_polygon_2d(num_points, num_polygons,
                                  num_points_per_polygon, points, polygons,
                                  vector, mask):
    inside = True
    slope_diff = 0
    for point_idx in range(num_points):  # N
        for polygon_idx in range(num_polygons):  # M
            inside = True
            for idx in range(num_points_per_polygon):  # 2
                #vector_slop = vector[polygon_idx, idx, 1] / [polygon_idx, idx, 0]
                #point_slop = (polygons[polygon_idx, idx, 1] - points[point_idx, 1]) / (polygons[polygon_idx, idx, 0] - points[point_idx, 0])
                slope_diff = (
                    polygons[polygon_idx, idx, 0] -
                    points[point_idx, 0]) * vector[polygon_idx, idx, 1]
                slope_diff -= (
                    polygons[polygon_idx, idx, 1] -
                    points[point_idx, 1]) * vector[polygon_idx, idx, 0]
                if slope_diff >= 0:
                    inside = False
                    break
            mask[point_idx, polygon_idx] = inside
    return mask


def points_in_convex_polygon_2d(points: np.ndarray,
                                polygons: np.ndarray,
                                clockwise: bool = True):
    # Convert polygons to directed vectors, the slope for each vector is vec_y / vec_x
    num_points = points.shape[0]  # [N, 2]
    num_polygons = polygons.shape[0]  # [M, 4, 2]
    num_points_per_polygon = polygons.shape[1]  # [M, 4, 2]

    if clockwise:
        vector = (
            polygons - polygons[:, [num_points_per_polygon - 1] +
                                list(range(num_points_per_polygon - 1)), :]
        )  # [M, 4, 2]
    else:
        vector = (
            polygons[:, [num_points_per_polygon - 1] +
                     list(range(num_points_per_polygon - 1)), :] - polygons
        )  # [M, 4, 2]

    mask = np.zeros((num_points, num_polygons), dtype='bool')
    mask = get_mask_points_in_polygon_2d(num_points, num_polygons,
                                         num_points_per_polygon, points,
                                         polygons, vector, mask)
    return mask


@numba.jit(nopython=True)
def corner_to_standup_nd_jit(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0])
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1])
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (
                                C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (
                                C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


def minmax_range_3d_to_corner_2d(minmax_range_3d: np.ndarray):
    center = minmax_range_3d[0:3]
    wlh = minmax_range_3d[3:6] - minmax_range_3d[0:3]
    data = np.asarray(np.hstack(
        (center, wlh,
         [0])))[np.newaxis,
                ...]  # add a fake orientation to construct a BBoxes3D
    bbox3d = BBoxes3D(data, origin=[0., 0., 0.])
    return bbox3d.corners_2d


@numba.jit(nopython=True)
def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j])**2 + (y1[i] - y1[j])**2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep


def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces


@numba.jit(nopython=True)
def surface_equ_3d_jit(surfaces):
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = np.zeros((num_polygon, max_num_surfaces, 3),
                          dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = sv0[1] * sv1[2] - sv0[2] * sv1[1]
            normal_vec[i, j, 1] = sv0[2] * sv1[0] - sv0[0] * sv1[2]
            normal_vec[i, j, 2] = sv0[0] * sv1[1] - sv0[1] * sv1[0]

            d[i, j] = (-surfaces[i, j, 0, 0] * normal_vec[i, j, 0] -
                       surfaces[i, j, 0, 1] * normal_vec[i, j, 1] -
                       surfaces[i, j, 0, 2] * normal_vec[i, j, 2])
    return normal_vec, d


def points_in_convex_polygon_3d_jit(points, polygon_surfaces,
                                    num_surfaces=None):
    """
    Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """

    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons, ), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces,
                                            normal_vec, d, num_surfaces)


@numba.njit
def _points_in_convex_polygon_3d_jit(points,
                                     polygon_surfaces,
                                     normal_vec,
                                     d,
                                     num_surfaces=None):
    """
    Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (points[i, 0] * normal_vec[j, k, 0] +
                        points[i, 1] * normal_vec[j, k, 1] +
                        points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def get_mask_of_points_in_bboxes3d(points, bboxes: BBoxes3D):
    corners_3d = bboxes.corners_3d
    surfaces = corner_to_surfaces_3d(corners_3d)
    mask = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return mask


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated 2D bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots = np.abs(BBoxes3D.limit_period(rots, 0.5, np.pi))
    cond = (rots > np.pi / 4)[..., np.newaxis]
    bboxes_center_dim = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    centers, dims = bboxes_center_dim[:, :2], bboxes_center_dim[:, 2:]
    bboxes = np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)

    return bboxes


def second_box_encode(boxes_3d, anchors):
    """
    Encode 3D bboxes for VoxelNet/PointPillars.
    Args:
        boxes_3d ([N, 7] np.ndarray): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] np.ndarray): anchors

    """
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    xg, yg, zg, wg, lg, hg, rg = np.split(boxes_3d, 7, axis=-1)

    diagonal = np.sqrt(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha

    lt = np.log(lg / la)
    wt = np.log(wg / wa)
    ht = np.log(hg / ha)

    rt = rg - ra
    return np.concatenate([xt, yt, zt, wt, lt, ht, rt], axis=-1)


def second_box_decode(encodings, anchors):
    """
    Decode 3D bboxes for VoxelNet/PointPillars.
    Args:
        encodings ([N, 7] np.ndarray): encoded boxes: x, y, z, w, l, h, r
        anchors ([N, 7] np.ndarray): anchors
    """
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    xt, yt, zt, wt, lt, ht, rt = np.split(encodings, 7, axis=-1)

    diagonal = np.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    lg = np.exp(lt) * la
    wg = np.exp(wt) * wa
    hg = np.exp(ht) * ha

    rg = rt + ra

    return np.concatenate([xg, yg, zg, wg, lg, hg, rg], axis=-1)


@numba.jit(nopython=True)
def iou_2d_jit(boxes, query_boxes, eps=0.0):
    """
    Calculate 2D box iou.

    Args:
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def rotation_3d_in_axis(points, angles, axis=0):
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def project_to_image(points_3d, proj_mat):
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.zeros(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def in_hull(p, hull):
    """
    param p: (N, K) test points
    param hull: (M, K) M corners of a box
    return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def boxes_to_corners_3d(boxes3d):
    """
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = np.array([
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    ]) / 2

    corners3d = np.tile(boxes3d[:, None, 3:6], (1, 8, 1)) * template[None, :, :]
    from paddle3d.geometries import PointCloud
    pointcloud_ = PointCloud(corners3d.reshape([-1, 8, 3]))
    pointcloud_.rotate_around_z(boxes3d[:, 6])
    corners3d = corners3d.reshape([-1, 8, 3])
    corners3d = corners3d + boxes3d[:, None, 0:3]

    return corners3d


def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_utils.py#L55

    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    mask = ((corners >= limit_range[0:3]) &
            (corners <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask


def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_utils.py#L91

    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    xyz_camera = boxes3d_camera[:, 0:3]
    l, h, w, r = boxes3d_camera[:, 3:
                                4], boxes3d_camera[:, 4:
                                                   5], boxes3d_camera[:, 5:
                                                                      6], boxes3d_camera[:,
                                                                                         6:
                                                                                         7]
    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_utils.py#L152

    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    xyz_lidar = boxes3d_lidar[:, 0:3]
    l, w, h, r = boxes3d_lidar[:, 3:
                               4], boxes3d_lidar[:, 4:
                                                 5], boxes3d_lidar[:, 5:
                                                                   6], boxes3d_lidar[:,
                                                                                     6:
                                                                                     7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_utils.py#L169

    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array(
        [l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2],
        dtype=np.float32).T
    z_corners = np.array(
        [w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.],
        dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([
            h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.
        ],
                             dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(
        ry.size, dtype=np.float32), np.ones(
            ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)], [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate(
        (x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
         z_corners.reshape(-1, 8, 1)),
        axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :,
                                                      0], rotated_corners[:, :,
                                                                          1], rotated_corners[:, :,
                                                                                              2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate(
        (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_utils.py#L215

    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(
            boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(
            boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(
            boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(
            boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image
