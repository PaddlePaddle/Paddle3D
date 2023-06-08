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

import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


def mean_col_by_row_with_offset_z(seg, offset_y, z):
    assert (len(seg.shape) == 2)

    center_ids = np.unique(seg[seg > 0])
    lines = []
    for idx, cid in enumerate(center_ids):  # 一个id
        cols, rows, z_val = [], [], []
        for y_op in range(seg.shape[0]):  # Every row
            condition = seg[y_op, :] == cid
            x_op = np.where(condition)[0]  # All pos in this row
            z_op = z[y_op, :]
            offset_op = offset_y[y_op, :]
            if x_op.size > 0:
                offset_op = offset_op[x_op]
                z_op = np.mean(z_op[x_op])
                z_val.append(z_op)
                x_op_with_offset = x_op + offset_op
                x_op = np.mean(x_op_with_offset)  # mean pos
                cols.append(x_op)
                rows.append(y_op + 0.5)
        lines.append((cols, rows, z_val))
    return lines


def bev_instance2points_with_offset_z(ids: np.ndarray,
                                      max_x=50,
                                      meter_per_pixal=(0.2, 0.2),
                                      offset_y=None,
                                      Z=None):
    center = ids.shape[1] / 2
    lines = mean_col_by_row_with_offset_z(ids, offset_y, Z)
    points = []
    # for i in range(1, ids.max()):
    for y, x, z in lines:  # cols, rows
        # x, y = np.where(ids == 1)
        x = np.array(x)[::-1]
        y = np.array(y)[::-1]
        z = np.array(z)[::-1]

        x = max_x / meter_per_pixal[0] - x
        y = y * meter_per_pixal[1]
        y -= center * meter_per_pixal[1]
        x = x * meter_per_pixal[0]

        y *= -1.0  # Vector is from right to left
        if len(x) < 2:
            continue
        spline = CubicSpline(x, y, extrapolate=False)
        points.append((x, y, z, spline))
    return points
