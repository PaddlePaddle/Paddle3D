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

from typing import Callable

import paddle

from pprndr.apis import manager
from pprndr.cameras.rays import Frustums, RayBundle, RaySamples
from pprndr.geometries import ContractionType
from pprndr.models.layers import OccupancyGrid
from pprndr.models.ray_samplers import BaseSampler
from pprndr.ray_marching import ray_marching, unpack_info

__all__ = ["VolumetricSampler"]


@manager.RAYSAMPLERS.add_component
class VolumetricSampler(BaseSampler):
    """
    Sampler that maintains an occupancy grid proposed in Instant-NGP.
    Reference:https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
    Skip empty space and early-stop when accumulated transmittance drops below a threshold.
    """

    def __init__(self,
                 occupancy_grid: OccupancyGrid = None,
                 grid_update_interval: int = 16,
                 step_size: float = 0.005,
                 scene_min_near: float = None,
                 scene_max_far: float = None,
                 cone_angle: float = 0.,
                 alpha_thresh: float = 0.):
        super(VolumetricSampler, self).__init__()

        self.occupancy_grid = occupancy_grid
        self.grid_update_interval = grid_update_interval
        self.scene_aabb = occupancy_grid.aabb if occupancy_grid.contraction_type == ContractionType.AABB else None
        self.step_size = step_size
        self.scene_min_near = scene_min_near
        self.scene_max_far = scene_max_far
        self.cone_angle = cone_angle
        self.alpha_thresh = alpha_thresh

    def get_sigma_fn(self,
                     origins: paddle.Tensor,
                     directions: paddle.Tensor,
                     density_fn: Callable = None) -> Callable:
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = paddle.index_select(origins, ray_indices, axis=0)
            t_directions = paddle.index_select(directions, ray_indices, axis=0)
            positions = t_origins + t_directions * (t_starts + t_ends) * 0.5
            return density_fn(positions)

        return sigma_fn

    @paddle.no_grad()
    def generate_ray_samples(self, ray_bundle: RayBundle,
                             density_fn: Callable) -> RaySamples:
        packed_info, starts, ends = ray_marching(
            origins=ray_bundle.origins,
            directions=ray_bundle.directions,
            scene_aabb=self.scene_aabb,
            occupancy_grid=self.occupancy_grid,
            sigma_fn=self.get_sigma_fn(
                ray_bundle.origins,
                ray_bundle.directions,
                density_fn=density_fn),
            step_size=self.step_size,
            min_near=self.scene_min_near,
            max_far=self.scene_max_far,
            stratified=self.training,
            cone_angle=self.cone_angle,
            alpha_thresh=self.alpha_thresh)

        ray_indices = unpack_info(packed_info)
        origins = paddle.index_select(ray_bundle.origins, ray_indices, axis=0)
        directions = paddle.index_select(
            ray_bundle.directions, ray_indices, axis=0)
        if ray_bundle.camera_ids is not None:
            camera_ids = paddle.index_select(
                ray_bundle.camera_ids, ray_indices, axis=0)
        else:
            camera_ids = None

        if ray_bundle.pixel_area is not None:
            pixel_area = paddle.index_select(
                ray_bundle.pixel_area, ray_indices, axis=0)
        else:
            pixel_area = None

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=starts,
                ends=ends,
                pixel_area=pixel_area),
            camera_ids=camera_ids,
            packed_info=packed_info,
            ray_indices=ray_indices)
        return ray_samples

    def forward(self,
                ray_bundle: RayBundle,
                *,
                cur_iter: int = None,
                density_fn: Callable = None,
                **kwargs) -> RaySamples:
        assert density_fn is not None, "density_fn is required for volumetric sampling"
        if self.training and cur_iter % self.grid_update_interval == 1 and self.occupancy_grid is not None:
            with paddle.no_grad():
                self.occupancy_grid.update(
                    cur_iter=cur_iter,
                    occ_eval_fn=lambda positions: density_fn(positions) * self.
                    step_size)
        return self.generate_ray_samples(ray_bundle, density_fn=density_fn)
