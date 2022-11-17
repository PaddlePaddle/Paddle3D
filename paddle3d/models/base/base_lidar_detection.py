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

from typing import List

import paddle
import paddle.nn as nn

from paddle3d.geometries import CoordMode
from paddle3d.models.base.base_detection import BaseDetectionModel


class BaseLidarModel(BaseDetectionModel):
    def __init__(self, box_with_velocity: bool = False):
        super().__init__(box_with_velocity=box_with_velocity)

    @property
    def input_spec(self) -> paddle.static.InputSpec:
        specs = [{
            "data":
            paddle.static.InputSpec(
                shape=[None, None], name='data', dtype='float32')
        }]
        return specs

    # @property
    # def input(self) -> List[dict]:
    #     input_list = [
    #         {'data' : [None, None]}
    #     ]

    @property
    def coord(self) -> CoordMode:
        return CoordMode.KittiLidar

    @property
    def sensor(self) -> str:
        return "lidar"
