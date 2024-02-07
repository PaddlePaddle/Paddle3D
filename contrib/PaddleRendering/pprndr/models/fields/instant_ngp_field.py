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

from typing import Dict, Tuple, Union

import paddle
import paddle.nn as nn

from pprndr.apis import manager
from pprndr.cameras.rays import RaySamples

try:
    import trunc_exp
except ModuleNotFoundError:
    from pprndr.cpp_extensions import trunc_exp

from pprndr.geometries.scene_box import ContractionType
from pprndr.models.fields import BaseDensityField
from pprndr.ray_marching import contract


@manager.FIELDS.add_component
class InstantNGPField(BaseDensityField):
    """
    Instant-NGP Field. Reference: https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf

    Args:
        dir_encoder: Direction encoder.
        pos_encoder: Position encoder.
        density_net: Density network.
        color_net: Color network.
        aabb: Scene aabb bounds of shape (6,), where
            aabb[:3] is the minimum (x,y,z) point.
            aabb[3:] is the maximum (x,y,z) point.
        contraction_type: Contraction type.
    """

    def __init__(self,
                 dir_encoder: nn.Layer,
                 pos_encoder: nn.Layer,
                 density_net: nn.Layer,
                 color_net: nn.Layer,
                 aabb: paddle.Tensor,
                 contraction_type: ContractionType = ContractionType.
                 UN_BOUNDED_SPHERE):
        super(InstantNGPField, self).__init__()

        self.aabb = paddle.to_tensor(aabb, dtype="float32")
        self.contraction_type = ContractionType(contraction_type)

        self.dir_encoder = dir_encoder
        self.pos_encoder = pos_encoder
        self.density_net = density_net
        self.color_net = color_net

    def get_density(self, ray_samples: Union[RaySamples, paddle.Tensor]
                    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if isinstance(ray_samples, RaySamples):
            positions = ray_samples.frustums.positions
        else:
            positions = ray_samples
        positions = contract(
            positions.reshape([-1, 3]), self.aabb, self.contraction_type)

        pos_embeddings = self.pos_encoder(positions)
        embeddings = self.density_net(pos_embeddings)

        density = trunc_exp.trunc_exp(embeddings[..., 0:1])
        geo_features = embeddings[..., 1:]

        return density, geo_features

    def get_outputs(self, ray_samples: RaySamples,
                    geo_features: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        dir_embeddings = self.dir_encoder(ray_samples.frustums.directions)
        embeddings = paddle.concat(
            [dir_embeddings.astype(geo_features.dtype), geo_features], axis=-1)

        color = self.color_net(embeddings).reshape(
            [*ray_samples.frustums.directions.shape[:-1], -1])

        return dict(rgb=color)
