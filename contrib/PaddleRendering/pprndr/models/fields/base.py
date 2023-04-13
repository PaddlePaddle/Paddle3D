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
from typing import Dict, Optional, Tuple, Union

import paddle
import paddle.nn as nn

from pprndr.cameras.rays import RaySamples

__all__ = ["BaseField"]


class BaseField(nn.Layer):
    @paddle.no_grad()
    def density_fn(self, positions: paddle.Tensor) -> paddle.Tensor:
        """
        Query densities at given positions in no_grad context.
        Used by ray marching process only.
        """
        is_training = self.training
        if is_training:
            self.eval()

        density = self.get_density(positions)[0]

        if is_training:
            self.train()

        return density

    @abc.abstractmethod
    def get_density(self, ray_samples: Union[RaySamples, paddle.Tensor]
                    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        """
        Query density of given ray samples. Returns a tensor of density and a tensor of geometry features.
        """

    @abc.abstractmethod
    def get_outputs(self, ray_samples: RaySamples,
                    geo_features: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        """
        Computes the final outputs (RGBs) of the field.
        """

    def forward(self, ray_samples: RaySamples) -> Dict[str, paddle.Tensor]:
        density, geo_features = self.get_density(ray_samples)
        outputs = self.get_outputs(ray_samples, geo_features=geo_features)
        outputs["density"] = density

        return outputs
