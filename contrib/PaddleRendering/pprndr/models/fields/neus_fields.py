#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Dict

import numpy as np
import paddle
import paddle.nn as nn

from pprndr.apis import manager
from pprndr.cameras.rays import RaySamples

try:
    import trunc_exp
except ModuleNotFoundError:
    from pprndr.cpp_extensions import trunc_exp

from pprndr.models.fields import BaseSDFField

__all__ = ["NeuSField"]


@manager.FIELDS.add_component
class NeuSField(BaseSDFField):
    """
    SDF and a rendering net
    """

    def __init__(self, pos_encoder: nn.Layer, view_encoder: nn.Layer,
                 sdf_network: nn.Layer, color_network: nn.Layer,
                 variance_init_val: float, scale: float):
        super(NeuSField, self).__init__()

        self.scale = scale

        # Encoder
        self.pos_encoder = pos_encoder
        self.view_encoder = view_encoder

        # Networks
        self.sdf_network = sdf_network
        self.color_network = color_network
        self.deviation_network = SingleVarianceNetwork(variance_init_val)

    def get_sdf_from_pts(self, pts: paddle.Tensor):
        # pts : (1, N, 3)
        pts = self.pos_encoder(pts)
        outputs = self.sdf_network(pts)
        signed_distances = outputs[:, :, :1]
        return signed_distances

    def get_sdf(self, ray_samples: RaySamples, compute_grad: bool = True):
        requires_grad = compute_grad or paddle.is_grad_enabled()
        with paddle.set_grad_enabled(requires_grad):
            if isinstance(ray_samples, RaySamples):
                pos_inputs = ray_samples.frustums.positions
            else:
                pos_inputs = ray_samples

            pos_inputs = pos_inputs * self.scale
            pos_inputs.stop_gradient = not requires_grad
            pos_embeddings = self.pos_encoder(pos_inputs)
            embeddings = self.sdf_network(pos_embeddings)
            signed_distances = embeddings[..., :1] / self.scale

            gradients = paddle.grad(
                outputs=signed_distances,
                inputs=pos_inputs,
                grad_outputs=paddle.ones_like(signed_distances),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0] if requires_grad else None

        sdf_features = embeddings[..., 1:]

        return signed_distances, (sdf_features, gradients)

    def get_outputs(self, ray_samples: RaySamples,
                    geo_features: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        positions = ray_samples.frustums.positions
        directions = ray_samples.frustums.directions

        embeddings, gradients = geo_features
        directions = self.view_encoder(directions)
        rendering_input = paddle.concat(
            [positions, directions, gradients, embeddings], axis=-1)
        color = self.color_network(rendering_input)

        return dict(rgb=color, gradients=gradients)

    def get_inside_filter_by_norm(
            self,
            ray_samples: RaySamples,
    ) -> paddle.Tensor:
        pts = ray_samples.frustums.positions
        pts_norm = paddle.linalg.norm(pts, p=2, axis=-1, keepdim=True)
        inside_sphere = (pts_norm < 1.0).astype('float32').detach()
        relax_inside_sphere = (pts_norm < 1.2).astype('float32').detach()
        return inside_sphere, relax_inside_sphere

    @property
    def inv_s(self):
        return self.deviation_network().clip(1e-6, 1e6)


class SingleVarianceNetwork(nn.Layer):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        variance = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Assign(
                np.array([init_val])))
        self.add_parameter('variance', variance)

    def forward(self):
        return paddle.exp(self.variance * 10.0)
