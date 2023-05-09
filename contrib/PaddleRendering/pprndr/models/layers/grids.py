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

from typing import Callable, List, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pprndr.apis import manager
from pprndr.geometries import ContractionType
from pprndr.ray_marching import contract_inv, grid_query

__all__ = ["OccupancyGrid"]


@manager.LAYERS.add_component
class OccupancyGrid(nn.Layer):
    def __init__(self,
                 aabb: Union[List, paddle.Tensor],
                 num_dim: int = 3,
                 resolution: Union[int, List[int], paddle.Tensor] = 128,
                 contraction_type: ContractionType = ContractionType.AABB,
                 warmup_steps: int = 256,
                 occupancy_thresh: float = .01,
                 ema_decay: float = .95):
        super(OccupancyGrid, self).__init__()

        if isinstance(resolution, int):
            self.resolution = paddle.to_tensor(
                [resolution] * num_dim, dtype="int32")
        else:
            self.resolution = paddle.to_tensor(resolution, dtype="int32")
        self.num_voxels = int(self.resolution.prod().item())

        self.aabb = paddle.to_tensor(aabb, dtype="float32").flatten()

        binary = paddle.zeros([self.num_voxels], dtype="bool")
        self.register_buffer("_binary", binary, persistable=True)

        occupancies = paddle.zeros([self.num_voxels])
        self.register_buffer("occupancies", occupancies, persistable=True)

        self.grid_coords = paddle.stack(
            paddle.meshgrid([paddle.arange(res) for res in self.resolution]),
            axis=-1).astype("float32").reshape([self.num_voxels, -1])
        self.grid_indices = paddle.arange(self.num_voxels)

        self.contraction_type = ContractionType(contraction_type)
        self.warmup_steps = int(warmup_steps)
        self.occupancy_thresh = occupancy_thresh
        self.ema_decay = ema_decay
        self.num_dim = num_dim

    @property
    def binary(self):
        return self._binary.reshape(self.resolution)

    @paddle.no_grad()
    def query_occ(self, samples: paddle.Tensor) -> paddle.Tensor:
        return grid_query(samples, self.aabb, self.binary,
                          self.contraction_type)

    @paddle.no_grad()
    def sample_uniform_and_occupied_voxels(self, n: int) -> paddle.Tensor:
        n = min(n, self.num_voxels)
        # sample n unoccupied voxels uniformly
        uniform_indices = paddle.nonzero(~self._binary).flatten()
        if n < len(uniform_indices):
            selector = paddle.randperm(len(uniform_indices))[:n]
            uniform_indices = uniform_indices[selector]
        # sample n occupied voxels uniformly
        occupied_indices = paddle.nonzero(self._binary).flatten()
        if n < len(occupied_indices):
            selector = paddle.randperm(len(occupied_indices))[:n]
            occupied_indices = occupied_indices[selector]
        indices = paddle.concat([uniform_indices, occupied_indices], axis=0)
        return indices

    @paddle.no_grad()
    def update(self, cur_iter: int, occ_eval_fn: Callable):
        if cur_iter < self.warmup_steps:
            indices = self.grid_indices
        else:
            indices = self.sample_uniform_and_occupied_voxels(
                self.num_voxels // 4)

        grid_coords = paddle.index_select(self.grid_coords, indices, axis=0)
        x = (grid_coords + paddle.rand(
            grid_coords.shape)) / self.resolution.astype("float32")
        if self.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            # only the points inside the sphere are valid
            mask = (x - 0.5).norm(axis=1) < 0.5

            # skip if no coordinate satisfies
            if not paddle.any(mask):
                return

            x = x[mask]
            indices = indices[mask]

        # voxel coordinates [0, 1]^3 -> world
        x = contract_inv(
            x,
            aabb=self.aabb,
            contraction_type=self.contraction_type,
        )
        occ = occ_eval_fn(x).squeeze(-1)

        # ema update
        paddle.scatter_(
            self.occupancies, indices,
            paddle.maximum(
                paddle.index_select(self.occupancies, indices, axis=0) *
                self.ema_decay, occ))
        self._binary = self.occupancies > paddle.clip(
            self.occupancies.mean(), max=self.occupancy_thresh)

    @paddle.no_grad()
    def upsample(self, resolution: int):
        occupancies = self.occupancies.reshape(
            self.resolution).unsqueeze(axis=[0, 1])

        occupancies = F.interpolate(
            occupancies,
            size=(resolution, resolution, resolution),
            mode='trilinear',
            align_corners=True,
            data_format='NCDHW')
        self.register_buffer(
            "occupancies", occupancies.flatten(), persistable=True)

        self._binary = self.occupancies > paddle.clip(
            self.occupancies.mean(), max=self.occupancy_thresh)

        self.resolution = paddle.to_tensor(
            [resolution] * self.num_dim, dtype="int32")
        self.num_voxels = int(self.resolution.prod().item())

        self.grid_coords = paddle.stack(
            paddle.meshgrid([paddle.arange(res) for res in self.resolution]),
            axis=-1).astype("float32").reshape([self.num_voxels, -1])
        self.grid_indices = paddle.arange(self.num_voxels)

    def forward(self, *inputs, **kwargs):
        pass
