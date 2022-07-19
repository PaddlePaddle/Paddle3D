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

import numpy as np


def random_flip_horizontal(image, depth_map, gt_boxes, calib):
    """
    This function refers to https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/datasets/augmentor/image_augmentor_utils.py#L5
    Performs random horizontal flip augmentation
    Args:
        image [np.ndarray(H_image, W_image, 3)]: Image
        depth_map [np.ndarray(H_depth, W_depth]: Depth map
        gt_boxes [np.ndarray(N, 7)]: 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib [calibration.Calibration]: Calibration object
    Returns:
        aug_image [np.ndarray(H_image, W_image, 3)]: Augmented image
        aug_depth_map [np.ndarray(H_depth, W_depth]: Augmented depth map
        aug_gt_boxes [np.ndarray(N, 7)]: Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
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

    return aug_image, aug_depth_map, aug_gt_boxes


def random_image_flip(data_dict=None):
    """
    This function refers to https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/datasets/augmentor/data_augmentor.py#L85
    """
    if data_dict is None:
        return
    images = data_dict["images"]
    depth_maps = data_dict["depth_maps"]
    gt_boxes = data_dict['gt_boxes']
    calib = data_dict["calib"]
    images, depth_maps, gt_boxes = random_flip_horizontal(
        images,
        depth_maps,
        gt_boxes,
        calib,
    )

    data_dict['images'] = images
    data_dict['depth_maps'] = depth_maps
    data_dict['gt_boxes'] = gt_boxes
    return data_dict
