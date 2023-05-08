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
import paddle.nn.functional as F

from pprndr.apis import manager
from pprndr.cameras.rays import RaySamples

try:
    import sh_encoder
except ModuleNotFoundError:
    from pprndr.cpp_extensions import sh_encoder

from pprndr.geometries import ContractionType
from pprndr.models.fields import BaseDensityField

__all__ = ["PlenoxelGrid"]


@manager.FIELDS.add_component
class PlenoxelGrid(BaseDensityField):
    def __init__(self,
                 radius: float = 1.3,
                 sh_degree: int = 2,
                 initial_resolution: int = 256,
                 prune_threshold: float = 0.001):
        super(PlenoxelGrid, self).__init__()

        assert sh_degree <= 4, "sh_degree must be <= 4"

        self.radius = paddle.to_tensor(radius, dtype="float32")
        self.aabb = paddle.to_tensor(
            [-radius, -radius, -radius, radius, radius, radius],
            dtype="float32").flatten()

        self.sh_degree = sh_degree
        self.sh_dim = (sh_degree + 1)**2
        self.sh_coeffs = self.create_parameter(
            [initial_resolution**3, self.sh_dim, 3],
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.0),
                learning_rate=150 * (initial_resolution**1.75)),
            dtype="float32")
        self.densities = self.create_parameter(
            [initial_resolution**3],
            attr=paddle.ParamAttr(
                initializer=nn.initializer.Constant(0.1),
                learning_rate=51.5 * (initial_resolution**2.37)),
            dtype="float32")

        grid_ids = paddle.arange(initial_resolution**3).reshape(
            [initial_resolution] * 3)
        self.register_buffer("grid_ids", grid_ids, persistable=True)

        self._resolution = paddle.to_tensor(initial_resolution, dtype="int32")
        self._voxel_size = self.radius * 2 / self._resolution

        self.prune_threshold = prune_threshold
        self.contraction_type = ContractionType.AABB

    @property
    def resolution(self):
        return self._resolution

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def binary(self):
        return (self.densities >= self.prune_threshold).reshape(
            [self.resolution] * 3)

    def _get_neighbors(self, positions: paddle.Tensor
                       ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        neighbor_offsets = paddle.to_tensor(
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1],
             [1, -1, 1], [1, 1, -1], [1, 1, 1]],
            dtype="float32") * self.voxel_size / 2.0  # [8, 3]
        neighbors = paddle.clip(
            positions.unsqueeze(-2) + neighbor_offsets.unsqueeze(0),
            -self.radius, self.radius)  # [N, 8, 3]
        neighbor_centers = paddle.clip(
            (paddle.floor(neighbors / self.voxel_size + 1e-5) + 0.5) *
            self.voxel_size, -self.radius + self.voxel_size / 2,
            self.radius - self.voxel_size / 2)  # [N, 8, 3]
        neighbor_indices = (
            paddle.floor(neighbor_centers / self.voxel_size + 1e-5)
            + self.resolution / 2.0).astype("int32").clip(
                0, self.resolution - 1)  # [N, 8, 3]

        return neighbor_centers, neighbor_indices

    def _lookup(self,
                indices: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        selected_ids = paddle.gather_nd(self.grid_ids, indices)  # [N, 8]
        empty_mask = selected_ids < 0.  # empty voxels have negative ids
        selected_ids = paddle.clip(selected_ids, min=0)

        neighbor_densities = paddle.gather_nd(self.densities,
                                              selected_ids[..., None])
        neighbor_densities[empty_mask] = 0.

        neighbor_sh_coeffs = paddle.gather_nd(self.sh_coeffs,
                                              selected_ids[..., None])
        neighbor_sh_coeffs[empty_mask] = 0.

        return neighbor_densities, neighbor_sh_coeffs

    def _get_trilinear_interp_weights(self, interp_offset):
        """
        interp_offset: [N, num_intersections, 3], the offset (as a fraction of voxel_len)
            from the first (000) interpolation point.
        """
        interp_offset_x = interp_offset[..., 0]  # [N]
        interp_offset_y = interp_offset[..., 1]  # [N]
        interp_offset_z = interp_offset[..., 2]  # [N]
        weight_000 = (1 - interp_offset_x) * (1 - interp_offset_y) * (
            1 - interp_offset_z)
        weight_001 = (1 - interp_offset_x) * (
            1 - interp_offset_y) * interp_offset_z
        weight_010 = (1 - interp_offset_x) * interp_offset_y * (
            1 - interp_offset_z)
        weight_011 = (1 - interp_offset_x) * interp_offset_y * interp_offset_z
        weight_100 = interp_offset_x * (1 - interp_offset_y) * (
            1 - interp_offset_z)
        weight_101 = interp_offset_x * (1 - interp_offset_y) * interp_offset_z
        weight_110 = interp_offset_x * interp_offset_y * (1 - interp_offset_z)
        weight_111 = interp_offset_x * interp_offset_y * interp_offset_z

        weights = paddle.stack([
            weight_000, weight_001, weight_010, weight_011, weight_100,
            weight_101, weight_110, weight_111
        ],
                               axis=-1)  # [N, 8]

        return weights

    def _trilinear_interpolation(
            self, positions: paddle.Tensor, neighbor_centers: paddle.Tensor,
            neighbor_densities: paddle.Tensor, neighbor_sh_coeffs: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        interp_offset = (
            positions - neighbor_centers[..., 0, :]) / self.voxel_size  # [N, 3]
        interp_weights = self._get_trilinear_interp_weights(
            interp_offset)  # [N, 8]

        densities = paddle.sum(
            interp_weights * neighbor_densities, axis=-1,
            keepdim=True)  # [N, 1]
        sh_coeffs = paddle.sum(
            interp_weights[..., None, None] * neighbor_sh_coeffs,
            axis=-3)  # [N, sh_dim, 3]

        return densities, sh_coeffs

    def get_density(self, ray_samples: Union[RaySamples, paddle.Tensor]
                    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if isinstance(ray_samples, RaySamples):
            positions = ray_samples.frustums.positions
        else:
            positions = ray_samples

        # Find the 8 neighbors of each positions
        neighbor_centers, neighbor_indices = self._get_neighbors(
            positions)  # [N, 8, 3], [N, 8, 3]

        # Look up neighbors' densities and SH coefficients
        neighbor_densities, neighbor_sh_coeffs = self._lookup(
            neighbor_indices)  # [N, 8], [N, 8, sh_dim, 3]

        # Tri-linear interpolation
        densities, geo_features = self._trilinear_interpolation(
            positions, neighbor_centers, neighbor_densities,
            neighbor_sh_coeffs)  # [N, 1], [N, sh_dim, 3]

        densities = F.relu(densities)

        return densities, geo_features

    def get_outputs(
            self,
            ray_samples: RaySamples,
            geo_features: paddle.Tensor,
    ) -> Dict[str, paddle.Tensor]:
        rays_d = ray_samples.frustums.directions
        dir_embeddings, _ = sh_encoder.sh_encode(
            rays_d, self.sh_degree, (not self.training)
            or rays_d.stop_gradient)  # [N, sh_dim]

        color = F.sigmoid(
            paddle.sum(geo_features * dir_embeddings.unsqueeze(-1),
                       axis=-2))  # [N, 3]

        return dict(rgb=color)
