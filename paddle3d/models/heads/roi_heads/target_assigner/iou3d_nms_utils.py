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
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""

import paddle

from paddle3d.ops import iou3d_nms_cuda


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).reshape([-1, 1])
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).reshape([-1, 1])
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).reshape([1, -1])
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).reshape([1, -1])

    # bev overlap
    overlaps_bev = iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a, boxes_b)

    max_of_min = paddle.maximum(boxes_a_height_min, boxes_b_height_min)
    min_of_max = paddle.minimum(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = paddle.clip(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).reshape([-1, 1])
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).reshape([1, -1])

    iou3d = overlaps_3d / paddle.clip(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d
