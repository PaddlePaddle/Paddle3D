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
import paddle.nn as nn
from PIL import Image


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


class GridMask(nn.Layer):
    """
    This class is modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/models/utils/grid_mask.py#L70
    """

    def __init__(self,
                 use_h,
                 use_w,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=0,
                 prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  #+ 1.#0.5

    def forward(self, x):
        #np.random.seed(0)
        if np.random.rand() > self.prob or not self.training:
            return x

        n, c, h, w = x.shape
        x = x.reshape([-1, h, w])
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        #np.random.seed(0)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        #np.random.seed(0)
        st_h = np.random.randint(d)
        #np.random.seed(0)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        #np.random.seed(0)
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 +
                    h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = paddle.to_tensor(mask, dtype=x.dtype)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            #np.random.seed(0)
            offset = paddle.to_tensor(
                2 * (np.random.rand(h, w) - 0.5), dtype=x.dtype)
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.reshape([n, c, h, w])
