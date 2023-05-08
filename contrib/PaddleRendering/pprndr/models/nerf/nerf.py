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

from pprndr.apis import manager
from pprndr.cameras.rays import RayBundle
from pprndr.models.fields import BaseDensityField
from pprndr.models.ray_samplers import BaseSampler
from pprndr.ray_marching import render_weight_from_density

__all__ = ["NeRF"]


@manager.MODELS.add_component
class NeRF(nn.Layer):
    def __init__(self,
                 coarse_ray_sampler: BaseSampler,
                 fine_ray_sampler: BaseSampler,
                 field: BaseDensityField,
                 rgb_renderer: nn.Layer,
                 coarse_rgb_loss: nn.Layer = nn.MSELoss(),
                 fine_rgb_loss: nn.Layer = nn.MSELoss()):
        super(NeRF, self).__init__()

        self.coarse_ray_sampler = coarse_ray_sampler
        self.fine_ray_sampler = fine_ray_sampler
        self.field = field
        self.rgb_renderer = rgb_renderer
        self.coarse_rgb_loss = coarse_rgb_loss
        self.fine_rgb_loss = fine_rgb_loss

    def _forward(self,
                 sample: Union[Tuple[RayBundle, dict], RayBundle],
                 cur_iter: int = None) -> Dict[str, paddle.Tensor]:
        if self.training:
            ray_bundle, pixel_batch = sample
        else:
            ray_bundle = sample

        coarse_samples = self.coarse_ray_sampler(
            ray_bundle, cur_iter=cur_iter, density_fn=self.field.density_fn)

        coarse_outputs = self.field(coarse_samples)
        coarse_weights = render_weight_from_density(
            t_starts=coarse_samples.frustums.starts,
            t_ends=coarse_samples.frustums.ends,
            densities=coarse_outputs["density"],
            packed_info=coarse_samples.packed_info)

        coarse_rgb, coarse_visibility_mask = self.rgb_renderer(
            coarse_outputs["rgb"],
            coarse_weights,
            ray_indices=coarse_samples.ray_indices,
            num_rays=ray_bundle.num_rays,
            return_visibility=self.training)

        fine_samples = self.fine_ray_sampler(
            ray_bundle=ray_bundle,
            ray_samples=coarse_samples,
            weights=coarse_weights)
        num_fine_samples = self.fine_ray_sampler.num_samples
        if getattr(self.fine_ray_sampler, "include_original", False):
            num_coarse_samples = self.coarse_ray_sampler.num_samples
            num_fine_samples = num_fine_samples + num_coarse_samples + 1

        fine_outputs = self.field(fine_samples)
        fine_weights = render_weight_from_density(
            t_starts=fine_samples.frustums.starts,
            t_ends=fine_samples.frustums.ends,
            densities=fine_outputs["density"],
            packed_info=fine_samples.packed_info)

        fine_rgb, fine_visibility_mask = self.rgb_renderer(
            fine_outputs["rgb"],
            fine_weights,
            ray_indices=fine_samples.ray_indices,
            num_rays=ray_bundle.num_rays,
            return_visibility=self.training)

        outputs = dict(rgb=fine_rgb)

        if self.training:
            coarse_rgb_loss = self.coarse_rgb_loss(
                coarse_rgb[coarse_visibility_mask],
                pixel_batch["pixels"][coarse_visibility_mask])
            fine_rgb_loss = self.fine_rgb_loss(
                fine_rgb[fine_visibility_mask],
                pixel_batch["pixels"][fine_visibility_mask])
            outputs["loss"] = dict(
                coarse_rgb_loss=.1 * coarse_rgb_loss,
                fine_rgb_loss=fine_rgb_loss)
            outputs["num_samples_per_batch"] = num_fine_samples

        return outputs

    def forward(self, *args, **kwargs):
        if hasattr(self, "amp_cfg_") and self.training:
            with paddle.amp.auto_cast(**self.amp_cfg_):
                return self._forward(*args, **kwargs)
        return self._forward(*args, **kwargs)
