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


class BaseMonoModel(BaseDetectionModel):
    def __init__(self,
                 box_with_velocity: bool = False,
                 need_camera_to_image: bool = True,
                 need_lidar_to_camera: bool = False,
                 need_down_ratios: bool = False,
                 image_height: Optional[int] = -1,
                 image_width: Optional[int] = -1):
        super().__init__(box_with_velocity=box_with_velocity)
        self.need_camera_to_image = need_camera_to_image
        self.need_lidar_to_camera = need_lidar_to_camera
        self.image_height = image_height
        self.image_width = image_width
        self.need_down_ratios = need_down_ratios

    @property
    def inputs(self) -> List[dict]:
        images = {
            'name': 'images',
            'dtype': 'float32',
            'shape': [1, 3, self.image_height, self.image_width]
        }
        res = [images]

        if self.need_camera_to_image:
            intrinsics = {
                'name': 'trans_cam_to_img',
                'dtype': 'float32',
                'shape': [1, 3, 4]
            }
            res.append(intrinsics)

        if self.need_lidar_to_camera:
            poses = {
                'name': 'trans_lidar_to_cam',
                'dtype': 'float32',
                'shape': [1, 4, 4]
            }
            res.append(poses)

        return res

    @property
    def sensor(self) -> str:
        return "camera"
