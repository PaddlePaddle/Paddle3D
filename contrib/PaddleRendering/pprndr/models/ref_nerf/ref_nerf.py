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
from pprndr.cameras.rays import RayBundle
from pprndr.models.fields import BaseDensityField
from pprndr.models.ray_samplers import BaseSampler
from pprndr.ray_marching import render_weight_from_density

__all__ = ["RefNeRF"]


@manager.MODELS.add_component
class RefNeRF(nn.Layer):
    def __init__(
            self,
            coarse_ray_sampler: BaseSampler,
            fine_ray_sampler: BaseSampler,
            field: BaseDensityField,
            rgb_renderer: nn.Layer,
            coarse_rgb_loss: nn.Layer = nn.MSELoss(),
            fine_rgb_loss: nn.Layer = nn.MSELoss(),
    ):

        super(RefNeRF, self).__init__()

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
            densities=coarse_outputs["density"])
        coarse_outputs["weights"] = coarse_weights.squeeze(-1)

        coarse_rgb, coarse_visibility_mask = self.rgb_renderer(
            coarse_outputs["rgb"],
            coarse_weights,
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
            densities=fine_outputs["density"])
        fine_outputs["weights"] = fine_weights.squeeze(-1)

        fine_rgb, fine_visibility_mask = self.rgb_renderer(
            fine_outputs["rgb"], fine_weights, return_visibility=self.training)

        outputs = dict(rgb=fine_rgb)

        if self.training:
            coarse_rgb_loss = self.coarse_rgb_loss(
                coarse_rgb[coarse_visibility_mask],
                pixel_batch["pixels"][coarse_visibility_mask])
            fine_rgb_loss = self.fine_rgb_loss(
                fine_rgb[fine_visibility_mask],
                pixel_batch["pixels"][fine_visibility_mask])

            coarse_orientation_loss = self._orientation_loss(
                coarse_outputs, ray_bundle.directions)
            fine_orientation_loss = self._orientation_loss(
                fine_outputs, ray_bundle.directions)

            coarse_pred_normal_loss = self._predicted_normal_loss(
                coarse_outputs)
            fine_pred_normal_loss = self._predicted_normal_loss(fine_outputs)

            outputs["loss"] = dict(
                coarse_rgb_loss=.1 * coarse_rgb_loss,
                fine_rgb_loss=fine_rgb_loss,
                coarse_orientation_loss=.01 * coarse_orientation_loss,
                fine_orientation_loss=.1 * fine_orientation_loss,
                coarse_pred_normal_loss=3e-5 * coarse_pred_normal_loss,
                fine_pred_normal_loss=3e-4 * fine_pred_normal_loss)

            outputs["num_samples_per_batch"] = num_fine_samples

        return outputs

    def _orientation_loss(self, rendered_result, directions):
        w = rendered_result["weights"]
        n = rendered_result["normals_pred"]
        if n is None:
            raise ValueError(
                "Normals cannot be None if orientation loss is on.")
        v = -1.0 * directions
        n_dot_v = (n * v[..., None, :]).sum(axis=-1)
        loss = paddle.mean((w * paddle.clip(n_dot_v, min=0.)**2).sum(axis=-1))

        return loss

    def _predicted_normal_loss(self, rendered_result):
        w = rendered_result["weights"]
        n = rendered_result["normals"]
        n_pred = rendered_result["normals_pred"]
        if n is None or n_pred is None:
            raise ValueError(
                "Predicted normals and gradient normals cannot be None if "
                "predicted normal loss is on.")
        loss = paddle.mean(
            (w * (1.0 - paddle.sum(n * n_pred, axis=-1))).sum(axis=-1))

        return loss

    def forward(self, *args, **kwargs):
        if hasattr(self, "amp_cfg_") and self.training:
            with paddle.amp.auto_cast(**self.amp_cfg_):
                return self._forward(*args, **kwargs)
        return self._forward(*args, **kwargs)
