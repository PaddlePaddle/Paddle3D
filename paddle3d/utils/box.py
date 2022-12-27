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

import numpy as np
import paddle


def boxes_iou_normal(boxes_a, boxes_b):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_utils.py#L238

    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = paddle.maximum(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = paddle.minimum(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = paddle.maximum(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = paddle.minimum(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = paddle.clip(x_max - x_min, min=0)
    y_len = paddle.clip(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / paddle.clip(
        area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_utils.py#L261

    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    rot_angle = paddle.abs(boxes3d[:, 6] -
                           paddle.floor(boxes3d[:, 6] / np.pi + 0.5) * np.pi)
    choose_dims = paddle.where(
        rot_angle[:, None] < np.pi / 4,
        paddle.gather(boxes3d, index=paddle.to_tensor([3, 4]), axis=1),
        paddle.gather(boxes3d, index=paddle.to_tensor([4, 3]), axis=1))
    aligned_bev_boxes = paddle.concat(
        [boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2],
        axis=1)
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_utils.py#L275

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)


def normalize_bbox(bboxes, pc_range):
    """
    This function is modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/util.py
    """
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.shape[-1] > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = paddle.concat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), axis=-1)
    else:
        normalized_bboxes = paddle.concat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), axis=-1)
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    """
    This function is modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/util.py
    """
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = paddle.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.shape[-1] > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = paddle.concat([cx, cy, cz, w, l, h, rot, vx, vy],
                                            axis=-1)
    else:
        denormalized_bboxes = paddle.concat([cx, cy, cz, w, l, h, rot], axis=-1)
    return denormalized_bboxes


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]

    if rows * cols == 0:
        return paddle.to_tensor(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    lt = paddle.maximum(bboxes1[..., :, None, :2],
                        bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = paddle.minimum(bboxes1[..., :, None, 2:],
                        bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = paddle.clip(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]

    if mode in ['iou', 'giou']:
        union = area1[..., None] + area2[..., None, :] - overlap
    else:
        union = area1[..., None]
    if mode == 'giou':
        enclosed_lt = paddle.minimum(bboxes1[..., :, None, :2],
                                     bboxes2[..., None, :, :2])
        enclosed_rb = paddle.maximum(bboxes1[..., :, None, 2:],
                                     bboxes2[..., None, :, 2:])

    eps = paddle.to_tensor([eps])
    union = paddle.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = paddle.clip(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = paddle.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
