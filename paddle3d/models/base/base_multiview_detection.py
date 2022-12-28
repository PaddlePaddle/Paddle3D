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

from typing import List, Optional

import paddle
import paddle.nn as nn

from paddle3d.models.base import BaseDetectionModel


class BaseMultiViewModel(BaseDetectionModel):
    def __init__(self,
                 box_with_velocity: bool = False,
                 num_cameras: int = 6,
                 need_timestamp: bool = False,
                 image_height: Optional[int] = -1,
                 image_width: Optional[int] = -1):
        super().__init__(box_with_velocity=box_with_velocity)
        self.num_cameras = num_cameras
        self.image_height = image_height
        self.image_width = image_width
        self.need_timestamp = need_timestamp

    @property
    def inputs(self) -> List[dict]:
        images = {
            'name': 'images',
            'dtype': 'float32',
            'shape':
            [1, self.num_cameras, 3, self.image_height, self.image_width]
        }
        res = [images]

        img2lidars = {
            'name': 'img2lidars',
            'dtype': 'float32',
            'shape': [1, self.num_cameras, 4, 4]
        }
        res.append(img2lidars)

        if self.need_timestamp:
            timestamps = {
                'name': 'timestamps',
                'dtype': 'float32',
                'shape': [1, self.num_cameras]
            }
            res.append(timestamps)

        return res

    @property
    def sensor(self) -> str:
        return "camera"
