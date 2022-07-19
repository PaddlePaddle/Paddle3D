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

import logging
import os
import pickle
import random
import shutil
import subprocess

import numpy as np
import paddle


def drop_info_with_name(info, name):
    """
    This function refers to https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/common_utils.py#L30
    """
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points = paddle.to_tensor(points)
    angle = paddle.to_tensor(angle)

    cosa = paddle.cos(angle)
    sina = paddle.sin(angle)
    zeros = paddle.zeros(shape=[points.shape[0]])
    ones = paddle.ones(shape=[points.shape[0]])
    rot_matrix = paddle.stack(
        [cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones],
        axis=1).reshape([-1, 3, 3]).cast("float32")
    points_rot = paddle.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = paddle.concat([points_rot, points[:, :, 3:]], axis=-1)
    return points_rot.numpy()


def mask_points_by_range(points, limit_range):
    """
    This function refers to https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/common_utils.py#L63
    """
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
        & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def keep_arrays_by_name(gt_names, used_classes):
    """
    This function refers to https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/common_utils.py#L130
    """
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds
