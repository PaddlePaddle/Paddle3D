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

import paddle


def create_meshgrid3d(depth,
                      height,
                      width,
                      normalized_coordinates=True,
                      dtype=None):
    """Generate a coordinate grid for an image.

    Args:
        depth: the image depth (channels).
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize

    Return:
        grid tensor with shape :math:`(1, D, H, W, 3)`.
    """
    xs = paddle.linspace(0, width - 1, width, dtype=dtype)
    ys = paddle.linspace(0, height - 1, height, dtype=dtype)
    zs = paddle.linspace(0, depth - 1, depth, dtype=dtype)
    # Fix TracerWarning
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
        zs = (zs / (depth - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid = paddle.stack(paddle.meshgrid([zs, xs, ys]), axis=-1)  # DxWxHx3
    return base_grid.transpose([0, 2, 1, 3]).unsqueeze(0)  # 1xDxHxWx3


def normalize_coords(coords, shape):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/grid_utils.py#L4

    Normalize coordinates of a grid between [-1, 1]
    Args:
        coords: Coordinates in grid
        shape: Grid shape [H, W]
    Returns:
        norm_coords: Normalized coordinates in grid
    """
    min_n = -1
    max_n = 1

    shape = paddle.flip(shape, axis=[0])  # Reverse ordering of shape
    # Subtract 1 since pixel indexing from [0, shape - 1]
    norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
    return norm_coords
