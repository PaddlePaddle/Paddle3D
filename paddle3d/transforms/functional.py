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
from typing import Tuple

import cv2
import numba
import numpy as np

from paddle3d.geometries.bbox import (box_collision_test, iou_2d_jit,
                                      rbbox2d_to_near_bbox)


def horizontal_flip(im: np.ndarray) -> np.ndarray:
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im


def vertical_flip(im: np.ndarray) -> np.ndarray:
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    return im


def normalize(im: np.ndarray, mean: Tuple[float, float, float],
              std: Tuple[float, float, float]) -> np.ndarray:
    im -= mean
    im /= std
    return im


def normalize_use_cv2(im: np.ndarray,
                      mean: np.ndarray,
                      std: np.ndarray,
                      to_rgb=True):
    """normalize an image with mean and std use cv2.
    """
    img = im.copy().astype(np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        # inplace
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    # inplace
    cv2.subtract(img, mean, img)
    # inplace
    cv2.multiply(img, stdinv, img)
    return img


def get_frustum(im_bbox, C, near_clip=0.001, far_clip=100):
    """
    Please refer to:
        <https://github.com/traveller59/second.pytorch/blob/master/second/core/box_np_ops.py#L521>
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array(
        [near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = im_bbox
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]], dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


@numba.jit(nopython=True)
def corner_to_surface_normal(corners):
    """
    Given coordinates the 3D bounding box's corners,
    compute surface normal located at each corner oriented towards inside the box.
    Please refer to:
        <https://github.com/traveller59/second.pytorch/blob/master/second/core/box_np_ops.py#L764>

    Args:
        corners (float array, [N, 8, 3]): Coordinates of 8 3d box corners.
    Returns:
        normals (float array, [N, 6, 4, 3]): Normals of 6 surfaces. Each surface is represented by 4 normals,
            located at the 4 corners.
    """
    num_boxes = corners.shape[0]
    normals = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_indices = np.array([
        0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7
    ]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                normals[i, j, k] = corners[i, corner_indices[j, k]]
    return normals


@numba.jit(nopython=True)
def points_to_voxel(points, voxel_size, point_cloud_range, grid_size, voxels,
                    coords, num_points_per_voxel, grid_idx_to_voxel_idx,
                    max_points_in_voxel, max_voxel_num):
    num_voxels = 0
    num_points = points.shape[0]
    # x, y, z
    coord = np.zeros(shape=(3, ), dtype=np.int32)

    for point_idx in range(num_points):
        outside = False
        for i in range(3):
            coord[i] = np.floor(
                (points[point_idx, i] - point_cloud_range[i]) / voxel_size[i])
            if coord[i] < 0 or coord[i] >= grid_size[i]:
                outside = True
                break
        if outside:
            continue
        voxel_idx = grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]]
        if voxel_idx == -1:
            voxel_idx = num_voxels
            if num_voxels >= max_voxel_num:
                continue
            num_voxels += 1
            grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]] = voxel_idx
            coords[voxel_idx, 0:3] = coord[::-1]
        curr_num_point = num_points_per_voxel[voxel_idx]
        if curr_num_point < max_points_in_voxel:
            voxels[voxel_idx, curr_num_point] = points[point_idx]
            num_points_per_voxel[voxel_idx] = curr_num_point + 1

    return num_voxels


def create_anchors_3d_stride(feature_size,
                             sizes=[1.6, 3.9, 1.56],
                             anchor_strides=[0.4, 0.4, 0.0],
                             anchor_offsets=[0.2, -39.8, -1.78],
                             rotations=[0, np.pi / 2]):
    """
    Generate 3D anchors according to specified strides.
    Please refer to:
        <https://github.com/traveller59/second.pytorch/blob/master/second/core/box_np_ops.py#L561>

    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    x_stride, y_stride, z_stride = anchor_strides
    x_offset, y_offset, z_offset = anchor_offsets
    z_centers = np.arange(feature_size[0], dtype=np.float32)
    y_centers = np.arange(feature_size[1], dtype=np.float32)
    x_centers = np.arange(feature_size[2], dtype=np.float32)
    z_centers = z_centers * z_stride + z_offset
    y_centers = y_centers * y_stride + y_offset
    x_centers = x_centers * x_stride + x_offset
    sizes = np.reshape(np.array(sizes, dtype=np.float32), [-1, 3])
    rotations = np.array(rotations, dtype=np.float32)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = sizes.shape[0]
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., None]
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


@numba.jit(nopython=True)
def sparse_sum_for_anchors_mask(coors, shape):
    ret = np.zeros(shape, dtype=np.float32)
    for i in range(coors.shape[0]):
        ret[coors[i, 1], coors[i, 2]] += 1
    return ret


@numba.jit(nopython=True)
def fused_get_anchors_area(dense_map, anchors_bv, stride, offset, grid_size):
    anchor_coor = np.zeros(anchors_bv.shape[1:], dtype=np.int32)
    grid_size_x = grid_size[0] - 1
    grid_size_y = grid_size[1] - 1
    N = anchors_bv.shape[0]
    ret = np.zeros((N, ), dtype=dense_map.dtype)
    for i in range(N):
        anchor_coor[0] = np.floor((anchors_bv[i, 0] - offset[0]) / stride[0])
        anchor_coor[1] = np.floor((anchors_bv[i, 1] - offset[1]) / stride[1])
        anchor_coor[2] = np.floor((anchors_bv[i, 2] - offset[0]) / stride[0])
        anchor_coor[3] = np.floor((anchors_bv[i, 3] - offset[1]) / stride[1])
        anchor_coor[0] = max(anchor_coor[0], 0)
        anchor_coor[1] = max(anchor_coor[1], 0)
        anchor_coor[2] = min(anchor_coor[2], grid_size_x)
        anchor_coor[3] = min(anchor_coor[3], grid_size_y)
        ID = dense_map[anchor_coor[3], anchor_coor[2]]
        IA = dense_map[anchor_coor[1], anchor_coor[0]]
        IB = dense_map[anchor_coor[3], anchor_coor[0]]
        IC = dense_map[anchor_coor[1], anchor_coor[2]]
        ret[i] = ID - IB - IC + IA
    return ret


@numba.jit(nopython=True)
def noise_per_box(bev_boxes, corners_2d, ignored_corners_2d, rotation_noises,
                  translation_noises):
    num_boxes = bev_boxes.shape[0]
    num_attempts = translation_noises.shape[1]

    selected_rotation_noises = np.zeros(num_boxes, dtype=rotation_noises.dtype)
    selected_translation_noises = np.zeros((num_boxes, 3),
                                           dtype=translation_noises.dtype)

    all_corners = np.concatenate((corners_2d, ignored_corners_2d), axis=0)

    for i in range(num_boxes):
        for j in range(num_attempts):
            # rotation
            current_corners = np.ascontiguousarray(corners_2d[i] -
                                                   bev_boxes[i, :2])
            rot_sin = np.sin(rotation_noises[i, j])
            rot_cos = np.cos(rotation_noises[i, j])
            rotation_matrix = np.array(
                [[rot_cos, -rot_sin], [rot_sin, rot_cos]], corners_2d.dtype)
            current_corners = current_corners @ rotation_matrix
            # translation
            current_corners += bev_boxes[i, :2] + translation_noises[i, j, :2]

            coll_mat = box_collision_test(
                current_corners.reshape(1, 4, 2), all_corners)
            coll_mat[0, i] = False
            if not coll_mat.any():
                # valid perturbation found
                selected_rotation_noises[i] = rotation_noises[i, j]
                selected_translation_noises[i] = translation_noises[i, j]
                break

    return selected_rotation_noises, selected_translation_noises


@numba.jit(nopython=True)
def perturb_object_points_(points, centers, point_masks, rotation_noises,
                           translation_noises):
    num_boxes = centers.shape[0]
    num_points = points.shape[0]
    rotation_matrices = np.zeros((num_boxes, 3, 3), dtype=points.dtype)
    for i in range(num_boxes):
        angle = rotation_noises[i]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        rotation_matrix = np.eye(3, dtype=points.dtype)
        rotation_matrix[0, 0] = rot_cos
        rotation_matrix[0, 1] = -rot_sin
        rotation_matrix[1, 0] = rot_sin
        rotation_matrix[1, 1] = rot_cos
        rotation_matrices[i] = rotation_matrix
    for i in range(num_points):
        for j in range(num_boxes):
            if point_masks[i, j] == 1:
                # rotation
                points[i, :3] -= centers[j, :3]
                points[i:i + 1, :3] = np.ascontiguousarray(
                    points[i:i + 1, :3]) @ rotation_matrices[j]
                points[i, :3] += centers[j, :3]
                # translation
                points[i, :3] += translation_noises[j]
                break


@numba.jit(nopython=True)
def perturb_object_bboxes_3d_(bboxes_3d, rotation_noises, translation_noises):
    bboxes_3d[:, 6] += rotation_noises
    bboxes_3d[:, :3] += translation_noises


def nearest_iou_similarity(bboxes_3d_1, bboxes_3d_2):
    """
    Compute similarity based on the squared distance metric.

    This function computes pairwise similarity between two BoxLists based on the
    negative squared distance metric.

    """

    boxes_bv_1 = rbbox2d_to_near_bbox(bboxes_3d_1[:, [0, 1, 3, 4, 6]])
    boxes_bv_2 = rbbox2d_to_near_bbox(bboxes_3d_2[:, [0, 1, 3, 4, 6]])
    return iou_2d_jit(boxes_bv_1, boxes_bv_2)


def random_depth_image_horizontal(data_dict=None):
    """
    Performs random horizontal flip augmentation
    Args:
    data_dict:
        image [np.ndarray(H_image, W_image, 3)]: Image
        depth_map [np.ndarray(H_depth, W_depth]: Depth map
        gt_boxes [np.ndarray(N, 7)]: 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib [calibration.Calibration]: Calibration object
    Returns:
    data_dict:
        aug_image [np.ndarray(H_image, W_image, 3)]: Augmented image
        aug_depth_map [np.ndarray(H_depth, W_depth]: Augmented depth map
        aug_gt_boxes [np.ndarray(N, 7)]: Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
    if data_dict is None:
        return
    image = data_dict["images"]
    depth_map = data_dict["depth_maps"]
    gt_boxes = data_dict['gt_boxes']
    calib = data_dict["calib"]

    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)
        aug_depth_map = np.fliplr(depth_map)

        # Flip 3D gt_boxes by flipping the centroids in image space
        aug_gt_boxes = copy.copy(gt_boxes)
        locations = aug_gt_boxes[:, :3]
        img_pts, img_depth = calib.lidar_to_img(locations)
        W = image.shape[1]
        img_pts[:, 0] = W - img_pts[:, 0]
        pts_rect = calib.img_to_rect(
            u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
        pts_lidar = calib.rect_to_lidar(pts_rect)
        aug_gt_boxes[:, :3] = pts_lidar
        aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]
    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes

    data_dict['images'] = aug_image
    data_dict['depth_maps'] = aug_depth_map
    data_dict['gt_boxes'] = aug_gt_boxes

    return data_dict


def blend_transform(img: np.ndarray, src_image: np.ndarray, src_weight: float,
                    dst_weight: float):
    """
    Transforms pixel colors with PIL enhance functions.
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = src_weight * src_image + dst_weight * img
        out = np.clip(img, 0, 255).astype(np.uint8)
    else:
        out = src_weight * src_image + dst_weight * img
    return out


def sample_point(sample, num_points):
    """ Randomly sample points by distance
    """
    if num_points == -1:
        return sample

    points = sample.data
    if num_points < len(points):
        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        choice = []
        if num_points > len(far_idxs_choice):
            near_idxs_choice = np.random.choice(
                near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            choice = np.random.choice(choice, num_points, replace=False)
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, len(points), dtype=np.int32)
        if num_points > len(points):
            extra_choice = np.random.choice(choice, num_points - len(points))
            choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)
    sample.data = sample.data[choice]

    return sample
