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

from typing import Dict, Tuple, Union, List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pprndr.apis import manager
from pprndr.cameras.rays import RaySamples
from pprndr.models.fields import BaseDensityField
from pprndr.models.encoders import SHEncoder, TensorCPEncoder, TensorVMEncoder, TriplaneEncoder

__all__ = ['TensoRFField']


@manager.FIELDS.add_component
class TensoRFField(BaseDensityField):
    """
    TensoRF Field. Reference: https://arxiv.org/abs/2003.08934

    Args:
        aabb: The aabb bounding box of the dataset.
        fea_encoder: The encoding method used for appearance encoding outputs.
        dir_encoder: The encoding method used for ray direction.
        density_encoder: The tensor encoding method used for scene density.
        color_encoder: The tensor encoding method used for scene color.
        color_head: Color network.
        appearance_dim: The number of dimensions for the appearance embedding.
        use_sh: Used spherical harmonics as the feature decoding function.
        sh_levels: The number of levels to use for spherical harmonics.
    """

    def __init__(self,
                 aabb: Union[paddle.Tensor, List],
                 fea_encoder: nn.Layer,
                 dir_encoder: nn.Layer,
                 density_encoder: nn.Layer,
                 color_encoder: nn.Layer,
                 color_head: nn.Layer,
                 appearance_dim: int = 27,
                 use_sh: bool = False,
                 sh_levels: int = 2):
        super(TensoRFField, self).__init__()

        self.aabb = paddle.to_tensor(aabb, dtype="float32").reshape([-1, 3])
        self.dir_encoder = dir_encoder
        self.fea_encoder = fea_encoder
        self.color_head = color_head
        self.density_encoder = density_encoder
        self.use_sh = use_sh
        self.color_encoder = color_encoder

        if self.use_sh:
            self.sh = SHEncoder(degree=sh_levels)
            self.B = nn.Linear(self.color_encoder.output_dim,
                               3 * self.sh.output_dim)
        else:
            self.B = nn.Linear(self.color_encoder.output_dim, appearance_dim)

        self.use_sh = use_sh

    def get_density(self, ray_samples: Union[RaySamples, paddle.Tensor]
                    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if isinstance(ray_samples, RaySamples):
            pos_inputs = ray_samples.frustums.positions
        else:
            pos_inputs = ray_samples
        positions = self.get_normalized_positions(pos_inputs)
        positions = positions * 2 - 1

        density = self.density_encoder(positions)
        density_enc = paddle.unsqueeze(paddle.sum(density, axis=-1), axis=-1)
        relu = nn.ReLU()
        density_enc = relu(density_enc)

        return density_enc, positions

    def density_L1(self):
        if isinstance(self.density_encoder, TensorCPEncoder):
            density_L1_loss = paddle.mean(paddle.abs(self.density_encoder.line_coef)) + \
                      paddle.mean(paddle.abs(self.color_encoder.line_coef))
        elif isinstance(self.density_encoder, TensorVMEncoder):
            density_L1_loss = paddle.mean(paddle.abs(self.density_encoder.line_coef)) + \
                      paddle.mean(paddle.abs(self.color_encoder.line_coef)) + \
                      paddle.mean(paddle.abs(self.density_encoder.plane_coef)) + \
                      paddle.mean(paddle.abs(self.color_encoder.plane_coef))
        elif isinstance(self.density_encoder, TriplaneEncoder):
            density_L1_loss = paddle.mean(paddle.abs(self.density_encoder.plane_coef)) + \
                      paddle.mean(paddle.abs(self.color_encoder.plane_coef))

        return density_L1_loss

    def get_outputs(self, ray_samples: RaySamples,
                    geo_features: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        positions = self.get_normalized_positions(
            ray_samples.frustums.positions)
        d = ray_samples.frustums.directions
        positions = positions * 2 - 1
        rgb_features = self.color_encoder(positions)
        rgb_features = self.B(rgb_features)

        d_encoded = self.dir_encoder(d)
        rgb_features_encoded = self.fea_encoder(rgb_features)

        if self.use_sh:
            sh_mult = paddle.unsqueeze(self.sh(d), axis=-1)
            rgb_sh = rgb_features.reshape([sh_mult.shape[0], -1, 3])
            color = F.relu(paddle.sum(sh_mult * rgb_sh, axis=-2) + 0.5)
        else:
            color = self.color_head(
                paddle.concat(
                    [rgb_features, d, rgb_features_encoded, d_encoded],
                    axis=-1))  # type: ignore
        return dict(rgb=color)

    def get_normalized_positions(self, positions: paddle.Tensor):
        """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box.

        Args:
            positions: the xyz positions
            aabb: the axis-aligned bounding box
        """
        aabb_lengths = self.aabb[1] - self.aabb[0]
        normalized_positions = (positions - self.aabb[0]) / aabb_lengths
        return normalized_positions

    def forward(self, ray_samples: RaySamples) -> Dict[str, paddle.Tensor]:
        density, _ = self.get_density(ray_samples)
        output = self.get_outputs(ray_samples, None)
        rgb = output["rgb"]

        return {"density": density, "rgb": rgb}
