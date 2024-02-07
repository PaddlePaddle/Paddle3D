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

import abc
from typing import Optional

import numpy as np

from paddle3d.apis import manager
from paddle3d.sample import Sample


class TransformABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, sample: Sample):
        """
        """


@manager.TRANSFORMS.add_component
class Compose(TransformABC):
    """
    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms

    def __call__(self, sample: Sample):
        """
        """
        for t in self.transforms:
            sample = t(sample)

        if sample.modality == 'image' and sample.meta.channel_order == 'hwc':
            sample.data = sample.data.transpose((2, 0, 1))
            sample.meta.channel_order = "chw"

        elif sample.modality == 'multimodal' or sample.modality == 'multiview':
            if 'img' in sample.keys():
                sample.img = np.stack(
                    [img.transpose(2, 0, 1) for img in sample.img], axis=0)

        return sample
