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

import paddle
import paddle.nn as nn

from pprndr.cameras import Cameras

__all__ = ["RayGenerator"]


class RayGenerator(nn.Layer):
    def __init__(self, cameras: Cameras, offset: float = 0.5):
        super(RayGenerator, self).__init__()

        self.cameras = cameras
        image_coords = cameras.get_image_coords(offset=offset)
        self.register_buffer("image_coords", image_coords)

    def forward(self, camera_ids: paddle.Tensor, pixel_indices: paddle.Tensor):
        """
        Generate rays according to ray indices.
        Args:
            camera_ids: [N] camera ids.
            pixel_indices: [N, 2], pixel indices, 2 = (row, col).
        """
        image_coords = paddle.gather_nd(self.image_coords, pixel_indices)

        ray_bundle = self.cameras.generate_rays(
            camera_ids=camera_ids, image_coords=image_coords)

        return ray_bundle
