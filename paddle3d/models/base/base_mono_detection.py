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

from paddle3d.geometries import CoordMode
from paddle3d.models.base import BaseDetectionModel


class BaseMonoModel(BaseDetectionModel):
    def __init__(self,
                 box_with_velocity: bool = False,
                 need_camera_intrinsic: bool = True,
                 need_camera_pose: bool = False,
                 image_height: Optional[int] = None,
                 image_width: Optional[int] = None):
        super().__init__(box_with_velocity=box_with_velocity)
        self.need_camera_intrinsic = need_camera_intrinsic
        self.need_camera_pose = need_camera_pose
        self.image_height = image_height
        self.image_width = image_width

    @property
    def inputs(self) -> List[dict]:
        """
        """
        images = {
            'name': 'camera.data',
            'dtype': 'float32',
            'shape': [1, 3, self.image_height, self.image_width]
        }
        res = [images]

        if self.need_camera_intrinsic:
            intrinsics = {
                'name': 'camera.intrinsic',
                'dtype': 'float32',
                'shape': [1, 3, 3]
            }
            res.append(intrinsics)

        if self.need_camera_pose:
            poses = {
                'name': 'camera.pose',
                'dtype': 'float32',
                'shape': [1, 3, 4]
            }
            res.append(poses)

        down_ratio = {'name': 'down_ratio', 'dtype': 'float32', 'shape': [1, 2]}
        res.append(down_ratio)
        return res

    # @property
    # def input_spec(self) -> paddle.static.InputSpec:
    #     image_spec = paddle.static.InputSpec(
    #         shape=[1, 3, self.image_height, self.image_width], dtype="float32")

    #     specs = [image_spec]

    #     if self.need_camera_intrinsic:
    #         intrinsic_spec = paddle.static.InputSpec(
    #             shape=[1, 3, 3], dtype="float32")
    #         specs.append(intrinsic_spec)

    #     if self.need_camera_pose:
    #         pose_spec = paddle.static.InputSpec(
    #             shape=[1, 3, 4], dtype="float32")
    #         specs.append(pose_spec)

    #     down_ratio_spec = paddle.static.InputSpec(shape=[1, 2], dtype="float32")
    #     specs.append(down_ratio_spec)
    #     return [specs]

    @property
    def coord(self) -> CoordMode:
        return CoordMode.KittiCamera

    @property
    def sensor(self) -> str:
        return "camera"
