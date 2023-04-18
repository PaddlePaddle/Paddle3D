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

from typing import Dict

import paddle

__all__ = ["PixelSampler"]


class PixelSampler(object):
    """
    Sample a batch of pixels from a batch of images. This happens on GPU.
    """

    def __init__(self, ray_batch_size: int):
        self._ray_batch_size = ray_batch_size

    @property
    def ray_batch_size(self):
        return self._ray_batch_size

    @ray_batch_size.setter
    def ray_batch_size(self, ray_batch_size):
        self._ray_batch_size = ray_batch_size

    def sample(self, image_batch: Dict) -> Dict:
        """
        Args:
        image_batch: A batch of images. A dict with keys:
            - id: Image IDs.
            - image: A tensor of shape [N, H, W, C].

        Returns:
        pixel_batch: A dict containing the following keys:
            - pixels: A tensor of shape [ray_batch_size, 3]
            - camera_ids: A tensor of shape [ray_batch_size], the global image (camera) ID of each pixel.
            - local_indices: A tensor of shape [ray_batch_size, 3], 3 = (idx wrt. current batch, row_idx, col_index)
        """
        im_batch_size, im_height, im_width, _ = image_batch['image'].shape

        indices = paddle.floor(
            paddle.rand((self._ray_batch_size, 3)) * paddle.to_tensor(
                [[im_batch_size, im_height, im_width]])).astype("int64")
        camera_indices, pixel_indices = paddle.split(indices, [1, 2], axis=1)

        pixels = paddle.gather_nd(image_batch['image'], indices)
        camera_ids = paddle.index_select(
            image_batch["camera_id"], camera_indices, axis=0)

        pixel_batch = dict(
            pixels=pixels, camera_ids=camera_ids, pixel_indices=pixel_indices)

        return pixel_batch
