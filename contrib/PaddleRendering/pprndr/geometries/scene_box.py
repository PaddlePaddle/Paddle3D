#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License")
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from enum import IntEnum
from typing import Union

import numpy as np
import paddle

__all__ = ["ContractionType", "SceneBox"]


class ContractionType(IntEnum):
    """
    Scene contraction options.

    Attributes:
        AABB: Linearly map the region of interest [x_0, x_1] to a
            unit cube in [0, 1].
        UN_BOUNDED_TANH: Contract an unbounded space into a unit cube in [0, 1]
            using tanh. The region of interest [x_0, x_1] is first mapped into
            [-0.5, +0.5] before applying tanh.
        UN_BOUNDED_SPHERE: Contract an unbounded space into a unit sphere. Used in
            Mip-Nerf 360: Unbounded Anti-Aliased Neural Radiance Fields.
    """

    AABB = 0
    UN_BOUNDED_TANH = 1
    UN_BOUNDED_SPHERE = 2


class SceneBox(object):
    """
    Data structure for scene bounding box.

    Args:
        aabb (np.ndarray): Scene aabb bounds of shape (2, 3), where
            aabb[0, :] is the minimum (x,y,z) point.
            aabb[1, :] is the maximum (x,y,z) point.
    """

    def __init__(self, aabb: np.ndarray):
        self.aabb = aabb

    @staticmethod
    def normalize_positions(positions: Union[np.ndarray, paddle.Tensor],
                            aabb: Union[np.ndarray, paddle.Tensor]
                            ) -> Union[np.ndarray, paddle.Tensor]:
        """
        Normalize positions to [0, 1].

        Args:
            positions (np.ndarray or paddle.Tensor): Positions to normalize.
            aabb (np.ndarray or paddle.Tensor): Scene aabb bounds of shape (6,), where
                aabb[:3] is the minimum (x,y,z) point.
                aabb[3:] is the maximum (x,y,z) point.

        Returns:
            Normalized positions.
        """
        min_xyz = aabb[:3]
        max_xyz = aabb[3:]

        return (positions - min_xyz) / (max_xyz - min_xyz)
