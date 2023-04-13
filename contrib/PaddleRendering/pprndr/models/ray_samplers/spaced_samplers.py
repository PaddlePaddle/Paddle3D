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
from pprndr.ray_marching import near_far_from_aabb

__all__ = ["SpacedSampler", "UniformSampler", "DisparitySampler", "LogSampler"]


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
                 num_samples: int = None,
                 aabb: Union[paddle.Tensor, Tuple, List] = None,
                 stratified: bool = True,
                 unified_jittering: bool = False):
        super(SpacedSampler, self).__init__(num_samples)

        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv
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
        bins = paddle.linspace(
            0.0, 1.0, num_samples + 1).unsqueeze_(0).repeat_interleave(
                num_rays, axis=0)  # [num_rays, num_samples + 1]

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
            nears, fars = near_far_from_aabb(ray_bundle.origins,
                                             ray_bundle.directions, self.aabb)
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
                 stratified: bool = True,
                 unified_jittering: bool = False):
        super(UniformSampler, self).__init__(
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            num_samples=num_samples,
            aabb=aabb,
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
                 stratified: bool = True,
                 unified_jittering: bool = False):
        super(DisparitySampler, self).__init__(
            spacing_fn=lambda x: 1.0 / x,
            spacing_fn_inv=lambda x: 1.0 / x,
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
