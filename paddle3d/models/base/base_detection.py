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

from paddle3d.models.base import Base3DModel

from typing import List


class BaseDetectionModel(Base3DModel):
    def __init__(self, box_with_velocity: bool = False):
        super().__init__()
        self.box_with_velocity = box_with_velocity

    @property
    def outputs(self) -> List[dict]:
        """Model output description."""

        boxdim = 7 if not self.box_with_velocity else 9
        box3ds = {'name': 'box3d', 'dtype': 'float32', 'shape': [-1, boxdim]}
        labels = {'name': 'label', 'dtype': 'int32', 'shape': [-1]}
        confidences = {'name': 'confidence', 'dtype': 'float32', 'shape': [-1]}
        return [box3ds, labels, confidences]
