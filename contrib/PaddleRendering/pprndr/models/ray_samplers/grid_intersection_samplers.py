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

from typing import Callable, Optional

import paddle

from pprndr.apis import manager
from pprndr.cameras.rays import Frustums, RayBundle, RaySamples
from pprndr.models.fields import PlenoxelGrid
from pprndr.models.ray_samplers import BaseSampler
from pprndr.ray_marching import ray_marching, unpack_info

__all__ = ["GridIntersectionSampler"]


@manager.RAYSAMPLERS.add_component
class GridIntersectionSampler(BaseSampler):
    """
    Sampler used in Plenoxels. Reference: https://arxiv.org/abs/2112.05131
    Sample intersections of rays and a voxel(Plenoxel) grid.
    """

    def __init__(self,
                 uniform: float = .5,
                 jitter: float = 0.,
                 alpha_thresh: float = 0.):
        super(GridIntersectionSampler, self).__init__()

        self.uniform = uniform
        self.jitter = jitter
        self.alpha_thresh = alpha_thresh

    def get_sigma_fn(self,
                     origins: paddle.Tensor,
                     directions: paddle.Tensor,
                     density_fn: Callable = None) -> Optional[Callable]:
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = paddle.index_select(origins, ray_indices, axis=0)
            t_directions = paddle.index_select(directions, ray_indices, axis=0)
            positions = t_origins + t_directions * t_starts
            return density_fn(positions)

        return sigma_fn

    @paddle.no_grad()
    def generate_ray_samples(self, ray_bundle: RayBundle,
                             plenoxel_grid: PlenoxelGrid) -> RaySamples:
        packed_info, starts, ends = ray_marching(
            origins=ray_bundle.origins,
            directions=ray_bundle.directions,
            scene_aabb=plenoxel_grid.aabb,
            occupancy_grid=plenoxel_grid,
            sigma_fn=self.get_sigma_fn(
                ray_bundle.origins,
                ray_bundle.directions,
                density_fn=plenoxel_grid.density_fn),
            alpha_thresh=self.alpha_thresh,
            step_size=float(self.uniform * plenoxel_grid.voxel_size),
            stratified=self.training)

        ray_indices = unpack_info(packed_info)
        origins = paddle.index_select(ray_bundle.origins, ray_indices, axis=0)
        directions = paddle.index_select(
            ray_bundle.directions, ray_indices, axis=0)
        if ray_bundle.camera_ids is not None:
            camera_ids = paddle.index_select(
                ray_bundle.camera_ids, ray_indices, axis=0)
        else:
            camera_ids = None

        positions = origins + starts * directions

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                positions=positions,
                starts=starts,
                ends=ends),
            camera_ids=camera_ids,
            packed_info=packed_info,
            ray_indices=ray_indices)

        return ray_samples
