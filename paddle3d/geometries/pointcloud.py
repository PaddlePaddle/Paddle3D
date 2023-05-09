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

from paddle3d.geometries.structure import _Structure


class PointCloud(_Structure):
    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 2 and data.ndim != 3:
            # When the data expands in 8 directions, the data.ndim is 3
            # [-1, 3] --> [-1, 8, 3]
            #   7 -------- 4
            #  /|         /|
            # 6 -------- 5 .
            # | |        | |
            # . 3 -------- 0
            # |/         |/
            # 2 -------- 1
            raise ValueError(
                'Illegal PointCloud data with number of dim {}'.format(
                    data.ndim))

        if data.shape[-1] < 3:
            raise ValueError('Illegal PointCloud data with shape {}'.format(
                data.shape))

    def scale(self, factor: float):
        """
        """
        self[..., :3] = self[..., :3] * factor

    def translate(self, translation: np.ndarray):
        """
        """
        self[..., :3] = self[..., :3] + translation

    def rotate_around_z(self, angle: np.ndarray):
        """
        """
        # Rotation matrix around the z-axis
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if self.ndim == 2:
            rotation_matrix = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=self.dtype)
        elif self.ndim == 3:
            zeros = np.zeros(self.shape[0])
            ones = np.ones(self.shape[0])
            rotation_matrix = np.array(
                [[rot_cos, -rot_sin, zeros], [rot_sin, rot_cos, zeros],
                 [zeros, zeros, ones]],
                dtype=self.dtype)
            rotation_matrix = rotation_matrix.reshape([-1, 3, 3])

        # Rotate x,y,z
        self[..., :3] = self[..., :3] @ rotation_matrix

    def flip(self, axis: int):
        """
        """
        if axis not in [0, 1]:
            raise ValueError(
                "Flip axis should be 0 or 1, but received is {}".format(axis))
        if axis == 0:  # flip along x-axis
            self[:, 1] = -self[:, 1]
        elif axis == 1:  # flip along y-axis
            self[:, 0] = -self[:, 0]

    def shuffle(self):
        self[...] = np.random.permutation(
            self[...])  # permutation is fater than shuffle

    def get_mask_of_points_outside_range(self, limit_range):
        mask = (self[:, 0] >= limit_range[0]) & (self[:, 0] <= limit_range[3]) \
            & (self[:, 1] >= limit_range[1]) & (self[:, 1] <= limit_range[4])
        return mask
