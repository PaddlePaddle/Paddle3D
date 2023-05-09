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

import paddle
import paddle.nn.functional as F

from pprndr.apis import manager
from pprndr.cameras.rays import Frustums, RayBundle, RaySamples
from pprndr.models.ray_samplers import BaseSampler
from pprndr.ray_marching import pdf_ray_marching, unpack_info

__all__ = ["PDFSampler", "EfficientPDFSampler"]


@manager.RAYSAMPLERS.add_component
class PDFSampler(BaseSampler):
    def __init__(self,
                 num_samples: int = None,
                 stratified: bool = True,
                 unified_jittering: bool = False,
                 include_original: bool = True,
                 weights_blur: bool = False,
                 histogram_padding: float = .01):
        super(PDFSampler, self).__init__(num_samples)

        self.stratified = stratified
        self.unified_jittering = unified_jittering
        self.include_original = include_original
        self.weights_blur = weights_blur
        self.histogram_padding = histogram_padding

    @paddle.no_grad()
    def generate_ray_samples(self,
                             ray_bundle: RayBundle,
                             ray_samples: RaySamples,
                             weights: paddle.Tensor = None,
                             num_samples: int = None,
                             **kwargs) -> RaySamples:
        """
        Generate ray samples according to a given distribution.

        Args:
            ray_bundle: Ray bundle to generate samples from.
            ray_samples: Existing ray samples.
            weights: Weights of each bin.
            num_samples: Number of samples to take along each ray.

        Returns:
            Positions and intervals for samples along a ray
        """

        num_samples = num_samples or self.num_samples
        assert num_samples is not None, "num_samples must be specified."
        if weights.ndim > 2:
            weights = weights.squeeze(-1)

        if self.weights_blur:
            weights_pad = paddle.concat(
                [weights[..., :1], weights, weights[..., -1:]], axis=-1)
            weights_max = paddle.fmax(weights_pad[..., :-1],
                                      weights_pad[..., 1:])
            weights = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        weights = weights + self.histogram_padding

        weights_sum = paddle.sum(weights, axis=-1, keepdim=True)
        padding = F.relu(1e-5 - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = paddle.minimum(paddle.ones_like(pdf), paddle.cumsum(pdf, axis=-1))
        cdf = paddle.concat([paddle.zeros_like(cdf[..., :1]), cdf], axis=-1)

        num_bins = num_samples + 1
        if self.stratified and self.training:
            # Stratified samples between 0 and 1
            u = paddle.linspace(
                0.0, 1.0 - (1.0 / num_bins), num=num_bins).expand(
                    (*cdf.shape[:-1], num_bins))
            if self.unified_jittering:
                jittering = paddle.rand((*cdf.shape[:-1], 1)) / num_bins
            else:
                jittering = paddle.rand(
                    (*cdf.shape[:-1], num_samples + 1)) / num_bins
            u += jittering
        else:
            # Uniform samples between 0 and 1
            u = paddle.linspace(0.0, 1.0 - (1.0 / num_bins), num=num_bins)
            u += 1.0 / (2.0 * num_bins)
            u = u.expand((*cdf.shape[:-1], num_bins))

        assert ray_samples.spacing2euclidean_fn is not None, "ray_samples.spacing2euclidean_fn must be provided"
        existing_bins = ray_samples.spacing_bins.squeeze(-1)

        inds = paddle.searchsorted(cdf, u, right=True)
        below = paddle.clip(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = paddle.clip(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = paddle.take_along_axis(cdf, below, axis=-1)
        bins_g0 = paddle.take_along_axis(existing_bins, below, axis=-1)
        cdf_g1 = paddle.take_along_axis(cdf, above, axis=-1)
        bins_g1 = paddle.take_along_axis(existing_bins, above, axis=-1)

        # if "paddle.nan_to_num" not working fine, replace the following with the this:
        #   > t = paddle.clip((u - cdf_g0) / (cdf_g1 - cdf_g0 + 1e-13), 0, 1)
        t = paddle.clip(
            paddle.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), nan=0.0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        if self.include_original:
            bins = paddle.sort(paddle.concat([existing_bins, bins], -1), -1)

        euclidean_bins = ray_samples.spacing2euclidean_fn(bins)
        ray_samples = ray_bundle.generate_ray_samples(
            euclidean_bins=euclidean_bins.unsqueeze(-1),
            spacing_bins=bins.unsqueeze(-1),
            spacing2euclidean_fn=ray_samples.spacing2euclidean_fn)

        return ray_samples


@manager.RAYSAMPLERS.add_component
class EfficientPDFSampler(BaseSampler):
    def __init__(self,
                 num_samples: int = None,
                 stratified: bool = True,
                 histogram_padding: float = .01):
        super(EfficientPDFSampler, self).__init__(num_samples)

        self.stratified = stratified
        self.histogram_padding = histogram_padding

    @paddle.no_grad()
    def generate_ray_samples(self,
                             ray_bundle: RayBundle,
                             ray_samples: RaySamples,
                             weights: paddle.Tensor = None,
                             num_samples: int = None) -> RaySamples:
        num_samples = num_samples or self.num_samples
        assert num_samples is not None, "num_samples must be specified."

        weights = weights + self.histogram_padding

        packed_info, starts, ends = pdf_ray_marching(
            starts=ray_samples.frustums.starts,
            ends=ray_samples.frustums.ends,
            weights=weights,
            packed_info=ray_samples.packed_info,
            num_samples=num_samples)

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
