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

from typing import Tuple

import numpy as np

from paddle3d.apis import manager
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC

__all__ = ["Normalize", "NormalizeRangeImage"]


@manager.TRANSFORMS.add_component
class Normalize(TransformABC):
    """
    """

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]):
        self.mean = mean
        self.std = std

        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))

        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample: Sample):
        """
        """
        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
        std = np.array(self.std)[:, np.newaxis, np.newaxis]

        if sample.modality == 'image':
            sample.data = sample.data.astype(np.float32, copy=False) / 255.0

            if sample.meta.channel_order != 'chw':
                mean = np.array(self.mean)
                std = np.array(self.std)

        sample.data = F.normalize(sample.data, mean, std)
        return sample


@manager.TRANSFORMS.add_component
class NormalizeRangeImage(TransformABC):
    """
    Normalize range image.

    Args:
        mean (list or tuple): Mean of range image.
        std (list or tuple): Standard deviation of range image.
    """

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]):
        if not (isinstance(mean,
                           (list, tuple)) and isinstance(std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))

        from functools import reduce
        if reduce(lambda x, y: x * y, std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

        self.mean = np.array(mean)[:, None, None]
        self.std = np.array(std)[:, None, None]

    def __call__(self, sample: Sample):
        """
        """
        sample.data = F.normalize(sample.data, self.mean, self.std)

        return sample
