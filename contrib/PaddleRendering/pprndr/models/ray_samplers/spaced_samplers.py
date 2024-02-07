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

from typing import List, Tuple, Union

import paddle

from pprndr.apis import manager
from pprndr.cameras.rays import RayBundle, RaySamples
from pprndr.models.ray_samplers import BaseSampler
from pprndr.ray_marching import near_far_from_aabb, near_far_from_sphere

__all__ = [
    "SpacedSampler", "UniformSampler", "DisparitySampler", "LogSampler",
    "NeuSOutsideSampler"
]


class SpacedSampler(BaseSampler):
    """
    Sample points along rays according to the spacing function.

    Args:
        spacing_fn: Function that map Euclidean ray distance t to a “normalized” ray distance s.
            (ie `lambda x : x` is uniform, `lambda x: 1.0 / x` is disparity).
        spacing_fn_inv: Inverse of spacing function.
        num_samples: Number of samples to take along each ray.
        aabb: Scene AABB bound.
        stratified: Whether to use stratified sampling during training. Default: True.
        unified_jittering: Whether to use the same jittering along each ray. Default: False.
    """

    def __init__(self,
                 spacing_fn: callable,
                 spacing_fn_inv: callable,
                 spacing_start: float = 0.0,
                 spacing_end: float = 1.0,
                 num_samples: int = None,
                 aabb: Union[paddle.Tensor, Tuple, List] = None,
                 compute_near_far_from_sphere: bool = False,
                 stratified: bool = True,
                 unified_jittering: bool = False):
        super(SpacedSampler, self).__init__(num_samples)

        self.spacing_start = spacing_start
        self.spacing_end = spacing_end
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv
        self.compute_near_far_from_sphere = compute_near_far_from_sphere
        self.aabb = paddle.to_tensor(
            aabb, dtype="float32").flatten() if aabb is not None else None
        self.stratified = stratified
        self.unified_jittering = unified_jittering

    @paddle.no_grad()
    def generate_ray_samples(self,
                             ray_bundle: RayBundle,
                             num_samples: int = None) -> RaySamples:
        num_samples = num_samples or self.num_samples
        assert num_samples is not None, "num_samples must be specified."
        num_rays = ray_bundle.num_rays
        bins = paddle.linspace(self.spacing_start, self.spacing_end,
                               num_samples + 1).unsqueeze_(0).repeat_interleave(
                                   num_rays,
                                   axis=0)  # [num_rays, num_samples + 1]

        if self.stratified and self.training:
            if self.unified_jittering:
                noises = paddle.rand([num_rays, 1])
            else:
                noises = paddle.rand([num_rays, num_samples + 1])
            bin_centers = (bins[:, :-1] + bins[:, 1:]) * 0.5
            bin_uppers = paddle.concat([bin_centers, bins[:, -1:]], axis=-1)
            bin_lowers = paddle.concat([bins[:, :1], bin_centers], axis=-1)
            bins = bin_lowers + (bin_uppers - bin_lowers) * noises

        if self.aabb is not None:
            assert (not self.compute_near_far_from_sphere)
            nears, fars = near_far_from_aabb(ray_bundle.origins,
                                             ray_bundle.directions, self.aabb)

        elif self.compute_near_far_from_sphere:
            assert (self.aabb is None)
            nears, fars = near_far_from_sphere(ray_bundle.origins,
                                               ray_bundle.directions)
        else:
            nears = paddle.zeros_like(ray_bundle.origins[:, 0]).unsqueeze_(-1)
            fars = paddle.full_like(ray_bundle.origins[:, 0],
                                    1e10).unsqueeze_(-1)

        s_nears = self.spacing_fn(nears)
        s_fars = self.spacing_fn(fars)
        spacing2euclidean_fn = lambda s: self.spacing_fn_inv((s_fars - s_nears)
                                                             * s + s_nears)

        euclidean_bins = spacing2euclidean_fn(
            bins)  # [num_rays, num_samples + 1]

        ray_samples = ray_bundle.generate_ray_samples(
            euclidean_bins=euclidean_bins.unsqueeze(-1),
            spacing_bins=bins.unsqueeze(-1),
            spacing2euclidean_fn=spacing2euclidean_fn)

        return ray_samples


@manager.RAYSAMPLERS.add_component
class NeuSOutsideSampler(BaseSampler):
    def __init__(self,
                 num_samples: int = None,
                 spacing_start: float = 1e-3,
                 spacing_end: float = 1.0,
                 aabb: Union[paddle.Tensor, Tuple, List] = None,
                 compute_near_far_from_sphere: bool = True,
                 stratified: bool = True,
                 unified_jittering: bool = False,
                 inside_interval_len: float = 1.0 / 128):
        self.spacing_start = spacing_start
        self.spacing_end = spacing_end
        self.stratified = stratified
        self.aabb = aabb
        self.compute_near_far_from_sphere = compute_near_far_from_sphere
        self.unified_jittering = unified_jittering
        self.inside_interval_len = inside_interval_len
        super(NeuSOutsideSampler, self).__init__(num_samples=num_samples)

    def generate_ray_samples(self,
                             ray_bundle: RayBundle,
                             num_samples: int = None):
        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_rays = ray_bundle.num_rays
        bins = paddle.linspace(self.spacing_start, self.spacing_end,
                               num_samples + 1).unsqueeze_(0).repeat_interleave(
                                   num_rays,
                                   axis=0)  # [num_rays, num_samples + 1]

        if self.stratified and self.training:
            if self.unified_jittering:
                noises = paddle.rand([num_rays, 1])
            else:
                noises = paddle.rand([num_rays, num_samples + 1])
            bin_centers = (bins[:, :-1] + bins[:, 1:]) * 0.5
            bin_uppers = paddle.concat([bin_centers, bins[..., -1:]], axis=-1)
            bin_lowers = paddle.concat([bins[..., :1], bin_centers], axis=-1)
            bins = bin_lowers + (bin_uppers - bin_lowers) * noises

        # Compute fars for calculating euclidean_bins
        if self.aabb is not None:
            assert (not self.compute_near_far_from_sphere)
            nears, fars = near_far_from_aabb(ray_bundle.origins,
                                             ray_bundle.directions, self.aabb)

        elif self.compute_near_far_from_sphere:
            assert (self.aabb is None)
            nears, fars = near_far_from_sphere(ray_bundle.origins,
                                               ray_bundle.directions)
        else:
            nears = paddle.zeros_like(ray_bundle.origins[:, 0]).unsqueeze_(-1)
            fars = paddle.full_like(ray_bundle.origins[:, 0],
                                    1e10).unsqueeze_(-1)

        assert (fars is not None)
        bins = bins[:, :-1]  # [num_rays, num_samples]
        euclidean_bins = fars / paddle.flip(
            bins, axis=-1) + self.inside_interval_len

        # Hard set positions
        origins = ray_bundle.origins.unsqueeze(-2).repeat_interleave(
            bins.shape[1], -2)
        directions = ray_bundle.directions.unsqueeze(-2).repeat_interleave(
            bins.shape[1], -2)

        positions = origins + directions * euclidean_bins.unsqueeze(-1)
        ray_samples = ray_bundle.generate_ray_samples(
            euclidean_bins=euclidean_bins.unsqueeze(-1),
            spacing_bins=bins.unsqueeze(-1),
            spacing2euclidean_fn=None,
            positions=positions)
        return ray_samples


@manager.RAYSAMPLERS.add_component
class UniformSampler(SpacedSampler):
    """
    Uniformly sample points along rays.
euclidean_bins
    Args:
        num_samples: Number of samples to take along each ray.
        aabb: Scene AABB bound.
        stratified: Whether to use stratified sampling during training. Default: True.
        unified_jittering: Whether to use the same jittering along each ray. Default: False.
    """

    def __init__(self,
                 num_samples: int = None,
                 aabb: paddle.Tensor = None,
                 compute_near_far_from_sphere: bool = False,
                 spacing_start: float = 0.0,
                 spacing_end: float = 1.0,
                 stratified: bool = True,
                 unified_jittering: bool = False):

        super(UniformSampler, self).__init__(
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            spacing_start=spacing_start,
            spacing_end=spacing_end,
            num_samples=num_samples,
            aabb=aabb,
            compute_near_far_from_sphere=compute_near_far_from_sphere,
            stratified=stratified,
            unified_jittering=unified_jittering)


@manager.RAYSAMPLERS.add_component
class DisparitySampler(SpacedSampler):
    """
    Sample points along rays linearly in disparity.

    Args:
        num_samples: Number of samples to take along each ray.
        aabb: Scene AABB bound.
        stratified: Whether to use stratified sampling during training. Default: True.
        unified_jittering: Whether to use the same jittering along each ray. Default: False.
    """

    def __init__(self,
                 num_samples: int = None,
                 aabb: paddle.Tensor = None,
                 spacing_start: float = 0.0,
                 spacing_end: float = 1.0,
                 stratified: bool = True,
                 unified_jittering: bool = False):
        super(DisparitySampler, self).__init__(
            spacing_fn=lambda x: 1.0 / x,
            spacing_fn_inv=lambda x: 1.0 / x,
            spacing_start=spacing_start,
            spacing_end=spacing_end,
            num_samples=num_samples,
            aabb=aabb,
            stratified=stratified,
            unified_jittering=unified_jittering)


@manager.RAYSAMPLERS.add_component
class LogSampler(SpacedSampler):
    """
    Sample points along rays according to logarithmic spacing.

    Args:
        num_samples: Number of samples to take along each ray.
        aabb: Scene AABB bound.
        stratified: Whether to use stratified sampling during training. Default: True.
        unified_jittering: Whether to use the same jittering along each ray. Default: False.
    """

    def __init__(self,
                 num_samples: int = None,
                 aabb: paddle.Tensor = None,
                 stratified: bool = True,
                 unified_jittering: bool = False):
        super(LogSampler, self).__init__(
            spacing_fn=paddle.log,
            spacing_fn_inv=paddle.exp,
            num_samples=num_samples,
            aabb=aabb,
            stratified=stratified,
            unified_jittering=unified_jittering)
