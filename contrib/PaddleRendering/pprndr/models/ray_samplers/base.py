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

import abc
from typing import Callable

import paddle.nn as nn

from pprndr.cameras.rays import RayBundle, RaySamples

__all__ = ["BaseSampler"]


class BaseSampler(nn.Layer):
    def __init__(self, num_samples: int = None):
        super(BaseSampler, self).__init__()

        self._num_samples = num_samples

    @property
    def num_samples(self):
        return self._num_samples

    @abc.abstractmethod
    def generate_ray_samples(self, ray_bundle: RayBundle,
                             **kwargs) -> RaySamples:
        """Generate ray samples."""

    def forward(self,
                ray_bundle: RayBundle,
                *,
                cur_iter: int = None,
                density_fn: Callable = None,
                **kwargs) -> RaySamples:
        return self.generate_ray_samples(ray_bundle, **kwargs)
