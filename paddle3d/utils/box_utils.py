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


def limit_period(val, offset: float = 0.5, period: float = np.pi):
    return val - np.floor(val / period + offset) * period


def boxes3d_kitti_lidar_to_lidar(boxes3d_lidar):
    """
    convert boxes from [x, y, z, w, l, h, yaw] to [x, y, z, l, w, h, heading], bottom_center -> obj_center
    """
    # yapf: disable
    w, l, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]
    boxes3d_lidar[:, 2] += h[:, 0] / 2
    # yapf: enable
    return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)],
                          axis=-1)


def boxes3d_lidar_to_kitti_lidar(boxes3d_lidar):
    """
    convert boxes from [x, y, z, l, w, h, heading] to [x, y, z, w, l, h, yaw], obj_center -> bottom_center
    """
    # yapf: disable
    l, w, h, heading = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:,6:7]
    boxes3d_lidar[:, 2] -= h[:, 0] / 2
    # yapf: enable
    return np.concatenate(
        [boxes3d_lidar[:, 0:3], w, l, h, -(heading + np.pi / 2)], axis=-1)
