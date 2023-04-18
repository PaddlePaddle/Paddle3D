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

from dataclasses import dataclass
from typing import Callable

import paddle

from pprndr.cameras.math_functionals import (Gaussians,
                                             conical_frustum_to_gaussian)

__all__ = ["RayBundle", "Frustums", "RaySamples"]


class Frustums(object):
    """
    Args:
        origins: Ray origins. Shape: [num_rays, num_samples, 3] or [num_total_samples, 3].
        directions: Ray directions. Shape: [num_rays, num_samples, 3] or [num_total_samples, 3].
        starts: Where each frustum starts along a ray. Shape: [num_rays, num_samples, 1] or [num_total_samples, 1].
        ends: Where each frustum ends along a ray. Shape: [num_rays, num_samples, 1] or [num_total_samples, 1].
        pixel_area: Pixel areas at a distance 1 from ray origins.
            Shape: [num_rays, num_samples, 1] or [num_total_samples, 1].
        positions: Coordinates of samples along the rays.
            Shape: [num_rays, num_samples, 3] or [num_total_samples, 3].
        offsets: Offsets for each sample position wrt. to the center of the frustum.
            Shape: [num_rays, num_samples, 3] or [num_total_samples, 3].
    """

    def __init__(self,
                 origins: paddle.Tensor,
                 directions: paddle.Tensor,
                 starts: paddle.Tensor,
                 ends: paddle.Tensor,
                 pixel_area: paddle.Tensor = None,
                 positions: paddle.Tensor = None,
                 offsets: paddle.Tensor = None):
        self.origins = origins
        self.directions = directions
        self.starts = starts
        self.ends = ends
        self.pixel_area = pixel_area
        self.offsets = offsets

        if positions is not None:
            self._positions = positions

    @property
    def deltas(self):
        return self.ends - self.starts

    @property
    def positions(self) -> paddle.Tensor:
        if hasattr(self, "_positions"):
            return self._positions
        else:
            positions = self.origins + self.directions * (
                self.starts + self.ends) / 2.0
            if self.offsets is not None:
                positions += self.offsets
            return positions

    @property
    def gaussians(self) -> Gaussians:
        """
        Calculates guassian approximation of conical frustum.
        Returns:
            Conical frustums approximated by gaussian distribution.
        """

        return conical_frustum_to_gaussian(
            origins=self.origins,
            directions=self.directions,
            starts=self.starts,
            ends=self.ends,
            pixel_area=self.pixel_area)


@dataclass
class RaySamples:
    frustums: Frustums
    """Frustums for each ray sample."""
    camera_ids: paddle.Tensor = None
    """Camera ids for each ray sample. Shape: [num_rays, num_samples, 1] or [num_total_samples, 1]"""
    spacing_bins: paddle.Tensor = None
    """Spacing bins for each ray sample. Only available if samples are not packed. Shape: [num_rays, num_samples + 1, 1]
    """
    spacing2euclidean_fn: Callable = None
    """Function to convert positions in spacing domain to euclidean domain. Only available if samples are not packed."""
    packed_info: paddle.Tensor = None
    """Packed info for each ray sample Only available if samples are packed. Shape: [num_rays, 2]"""
    ray_indices: paddle.Tensor = None
    """Ray indices for each ray sample. Only available if samples are packed. Shape: [num_total_samples, 1]"""

    @property
    def spacing_starts(self):
        return self.spacing_bins[..., :-1, :]

    @property
    def spacing_ends(self):
        return self.spacing_bins[..., 1:, :]


@dataclass
class RayBundle:
    origins: paddle.Tensor
    """Ray origins. Shape: [num_rays, 3]."""
    directions: paddle.Tensor
    """Ray directions. Shape: [num_rays, 3]."""
    pixel_area: paddle.Tensor
    """Pixel areas at a distance 1 from ray origins. Shape: [num_rays, 1]."""
    camera_ids: paddle.Tensor
    """Camera ids for each ray. Shape: [num_rays, 1]."""

    @property
    def num_rays(self):
        return len(self.origins)

    def __len__(self):
        return self.num_rays

    def __getitem__(self, indices) -> "RayBundle":
        return RayBundle(
            origins=self.origins[indices],
            directions=self.directions[indices],
            pixel_area=self.pixel_area[indices],
            camera_ids=self.camera_ids[indices])

    def generate_ray_samples(
            self,
            euclidean_bins: paddle.Tensor,
            spacing_bins: paddle.Tensor,
            spacing2euclidean_fn: Callable = None) -> RaySamples:
        n_smaples_per_ray = euclidean_bins.shape[-2] - 1

        ray_bundle = self[..., None, :]

        frustums = Frustums(
            origins=ray_bundle.origins.repeat_interleave(
                n_smaples_per_ray, axis=-2),
            directions=ray_bundle.directions.repeat_interleave(
                n_smaples_per_ray, axis=-2),
            starts=euclidean_bins[..., :-1, :],
            ends=euclidean_bins[..., 1:, :],
            pixel_area=ray_bundle.pixel_area.repeat_interleave(
                n_smaples_per_ray, axis=-2))
        ray_samples = RaySamples(
            frustums=frustums,
            camera_ids=ray_bundle.camera_ids.repeat_interleave(
                n_smaples_per_ray, axis=-2),
            spacing_bins=spacing_bins,
            spacing2euclidean_fn=spacing2euclidean_fn)
        return ray_samples
