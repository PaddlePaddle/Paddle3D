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

from typing import Dict, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pprndr.apis import manager
from pprndr.cameras.rays import RaySamples

try:
    import trunc_exp
except ModuleNotFoundError:
    from pprndr.cpp_extensions import trunc_exp

from pprndr.models.fields import BaseDensityField

__all__ = ['RefNeRFField']


@manager.FIELDS.add_component
class RefNeRFField(BaseDensityField):
    """
    Ref-NeRF Field. Reference: https://arxiv.org/abs/2112.03907

    Args:
        dir_encoder: Direction encoder.
        pos_encoder: Position encoder.
        density_head: Density network.
        rgb_head: RGB network.
        normal_head: Normal network.
        use_integrated_encoding: Used integrated samples as encoding input,
            as proposed in mip-NeRF (https://arxiv.org/abs/2103.13415).
    """

    def __init__(self,
                 dir_encoder: nn.Layer,
                 pos_encoder: nn.Layer,
                 stem_net: nn.Layer,
                 density_head: nn.Layer,
                 view_net: nn.Layer,
                 rgb_head: nn.Layer,
                 normal_head: nn.Layer,
                 rgb_diffuse_layer: nn.Layer,
                 tint_layer: nn.Layer,
                 roughness_layer: nn.Layer,
                 roughness_bias: float = None,
                 density_noise: float = None,
                 density_bias: float = None,
                 bottleneck_noise: float = None,
                 rgb_premultiplier: float = None,
                 rgb_bias: float = None,
                 rgb_padding: float = None,
                 use_integrated_encoding: bool = False,
                 bottleneck_width: int = 128):
        super(RefNeRFField, self).__init__()

        self.dir_encoder = dir_encoder
        self.pos_encoder = pos_encoder
        self.stem_net = stem_net
        self.density_head = density_head
        self.bottleneck_layer = nn.Linear(stem_net.output_dim, bottleneck_width)
        self.view_net = view_net
        self.rgb_head = rgb_head
        self.normal_head = normal_head
        self.rgb_diffuse_layer = rgb_diffuse_layer
        self.tint_layer = tint_layer
        self.roughness_layer = roughness_layer

        self.density_noise = density_noise
        self.density_bias = density_bias
        self.roughness_bias = roughness_bias
        self.bottleneck_noise = bottleneck_noise
        self.rgb_premultiplier = rgb_premultiplier
        self.rgb_bias = rgb_bias
        self.rgb_padding = rgb_padding
        self.use_integrated_encoding = use_integrated_encoding

    def _reflect(self, directions, normals):
        """Reflect view directions about normals.

        The reflection of a vector v about a unit vector n is a vector u such that
        dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
        equations is u = 2 dot(n, v) n - v.

        Args:
            directions: [..., 3] tensor of view directions.
            normals: [..., 3] tensor of normal directions (assumed to be unit vectors).

        Returns:
            [..., 3] tensor of reflection directions.
        """
        return (2.0 * paddle.sum(normals * directions, axis=-1, keepdim=True) *
                normals - directions)

    def _linear_to_srgb(self, linear, eps=1e-10):

        srgb0 = 323 / 25 * linear
        srgb1 = (211 * paddle.clip(linear, min=eps)**(5 / 12) - 11) / 200
        return paddle.where(linear <= 0.0031308, srgb0, srgb1)

    def get_density(self, ray_samples: RaySamples):

        with paddle.set_grad_enabled(True):
            if self.use_integrated_encoding:
                pos_inputs = ray_samples.frustums.gaussians
                pos_inputs.mean.stop_gradient = False
            else:
                pos_inputs = ray_samples.frustums.positions

            pos_embeddings = self.pos_encoder(pos_inputs)
            embeddings = self.stem_net(pos_embeddings)
            raw_density = self.density_head(embeddings)

            mean_grad = paddle.grad(
                outputs=raw_density.sum(),
                inputs=pos_inputs.mean,
                retain_graph=True)[0]

            normals = -F.normalize(mean_grad, p=2, axis=-1)

        if self.density_noise is not None:
            raw_density += paddle.randn(
                raw_density.shape, dtype=raw_density.dtype) * self.density_noise
        if self.density_bias is not None:
            raw_density += self.density_bias

        density = F.softplus(raw_density)

        return density, (embeddings, normals)

    def get_outputs(self, ray_samples: RaySamples,
                    geo_features: Union[paddle.Tensor, tuple]
                    ) -> Dict[str, paddle.Tensor]:

        directions = ray_samples.frustums.directions
        geo_feature, normals = geo_features

        grad_pred = self.normal_head(geo_feature)
        normals_pred = -F.normalize(grad_pred, p=2, axis=-1)
        normals_to_use = normals_pred

        raw_roughness = self.roughness_layer(geo_feature)
        if self.roughness_bias is not None:
            raw_roughness += self.roughness_bias
        roughness = F.softplus(raw_roughness)

        refdirs = self._reflect(-directions, normals_to_use)
        dir_enc = self.dir_encoder(refdirs, roughness)
        dotprod = paddle.sum(normals_to_use * directions, axis=-1, keepdim=True)

        bottleneck = self.bottleneck_layer(geo_feature)
        if self.bottleneck_noise is not None:
            bottleneck += self.bottleneck_noise * paddle.randn(bottleneck.shape)

        view_inputs = paddle.concat([bottleneck, dir_enc, dotprod], axis=-1)
        view_embeddings = self.view_net(view_inputs)
        raw_rgb = self.rgb_head(view_embeddings)

        if self.rgb_premultiplier is not None:
            raw_rgb = self.rgb_premultiplier * raw_rgb
        if self.rgb_bias is not None:
            raw_rgb += self.rgb_bias

        rgb = F.sigmoid(raw_rgb)
        raw_rgb_diffuse = self.rgb_diffuse_layer(geo_feature)
        tint = self.tint_layer(geo_feature)

        diffuse_linear = F.sigmoid(raw_rgb_diffuse -
                                   paddle.log(paddle.to_tensor([3.0])))
        specular_linear = tint * rgb
        rgb = paddle.clip(
            self._linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

        if self.rgb_padding is not None:
            rgb = rgb * (1. + 2. * self.rgb_padding) - self.rgb_padding

        return dict(
            rgb=rgb,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
        )
