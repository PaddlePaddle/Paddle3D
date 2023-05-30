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
    def bin_points(self) -> paddle.Tensor:
        # TODO: [Warning] If bin_points is passed to networks,
        #  note that there is a mismatch between the shapes of bin_points and shape of directions.
        if self.origins.ndim == 3:
            origins = paddle.concat([self.origins, self.origins[:, -1:, :]],
                                    axis=-2)
            directions = paddle.concat(
                [self.directions, self.directions[:, -1:, :]], axis=-2)
            points = origins + directions * paddle.concat(
                [self.starts, self.ends[:, -1:, :]],
                axis=-2)  # [num_rays, num_samples + 1, 3]
        else:
            origins = paddle.repeat_interleave(self.origins, repeats=2, axis=-2)
            directions = paddle.repeat_interleave(
                self.directions, repeats=2, axis=-2)
            intervals = paddle.stack([self.starts, self.ends], axis=-2).flatten(
                -3, -2)
            points = origins + directions * intervals  # [num_total_samples * 2, 3]

        return points

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

    def merge_samples(self,
                      sample_list,
                      ray_bundle=None,
                      sample_dist=2. / 128.,
                      mode='force_concat'):
        if len(sample_list) == 0:
            raise RuntimeError("Empty sample list.")

        if mode == "sort":
            # spacing2euclidean_fn must be the same one.
            for sample in sample_list:
                assert (
                    self.spacing2euclidean_fn == sample.spacing2euclidean_fn)
            bins = self.spacing_bins[:, :-1].squeeze(-1)

            for i, sample in enumerate(sample_list):
                if i != len(sample_list) - 1:
                    bins = paddle.concat(
                        [bins, sample.spacing_bins[:, :-1].squeeze(-1)],
                        axis=-1)
                else:
                    bins = paddle.concat(
                        [bins, sample.spacing_bins.squeeze(-1)], axis=-1)

            bins = paddle.sort(bins, axis=-1)
            index = paddle.argsort(bins, axis=-1)
            euclidean_bins = self.spacing2euclidean_fn(bins)
            ray_samples = ray_bundle.generate_ray_samples(
                euclidean_bins=euclidean_bins.unsqueeze(-1),
                spacing_bins=bins.unsqueeze(-1),
                spacing2euclidean_fn=self.spacing2euclidean_fn)
            return ray_samples, index

        elif mode == 'force_concat':
            # NOTE: This func. concatenates frustums, spacing_bins and spacing2euclidean_fn are disabled.
            euclidean_bins = paddle.concat(
                [self.frustums.starts, self.frustums.ends[:, -1:, :]], axis=1)
            origins = self.frustums.origins
            directions = self.frustums.directions
            pixel_area = self.frustums.pixel_area
            camera_ids = self.camera_ids

            for i, sample in enumerate(sample_list):
                origin_pad = sample.frustums.origins[:, -1:, :]
                origins = paddle.concat(
                    [origins, sample.frustums.origins, origin_pad], axis=-2)

                direction_pad = sample.frustums.directions[:, -1:, :]
                directions = paddle.concat(
                    [directions, sample.frustums.directions, direction_pad],
                    axis=-2)
                pixel_area = paddle.concat(
                    [pixel_area, sample.frustums.pixel_area], axis=-2)

                camera_ids = paddle.concat([camera_ids, sample.camera_ids],
                                           axis=-2)
                sample_euclidean_bins = paddle.concat(
                    [sample.frustums.starts, sample.frustums.ends[:, -1:, :]],
                    axis=1)
                euclidean_bins = paddle.concat(
                    [euclidean_bins, sample_euclidean_bins], axis=1)

            starts = euclidean_bins[:, :-1, :]
            ends = euclidean_bins[:, 1:, :]

            merged_frustums = Frustums(
                origins=origins,
                directions=directions,
                starts=starts,
                ends=ends,
                pixel_area=pixel_area,
                offsets=None,
            )
            ray_samples = RaySamples(
                frustums=merged_frustums,
                camera_ids=camera_ids,
                spacing_bins=None,
                spacing2euclidean_fn=None)
            return ray_samples
        else:
            raise NotImplementedError(
                "[mode] should be either 'force_concat' or 'sort'")


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

    def generate_ray_samples(self,
                             euclidean_bins: paddle.Tensor,
                             spacing_bins: paddle.Tensor,
                             spacing2euclidean_fn: Callable = None,
                             positions: paddle.Tensor = None) -> RaySamples:
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
                n_smaples_per_ray, axis=-2),
            positions=positions)
        ray_samples = RaySamples(
            frustums=frustums,
            camera_ids=ray_bundle.camera_ids.repeat_interleave(
                n_smaples_per_ray, axis=-2),
            spacing_bins=spacing_bins,
            spacing2euclidean_fn=spacing2euclidean_fn)
        return ray_samples
