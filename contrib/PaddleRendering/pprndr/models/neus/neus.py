#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Dict, Tuple, Optional

import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import numpy as np
import mcubes

from pprndr.apis import manager
from pprndr.cameras.rays import RayBundle
from pprndr.models.fields import BaseDensityField
from pprndr.models.ray_samplers import BaseSampler
from pprndr.ray_marching import render_alpha_from_densities, render_weights_from_alpha, render_alpha_from_sdf, \
    get_anneal_coeff, get_cosine_coeff
from pprndr.utils.logger import logger

__all__ = ["NeuS"]


@manager.MODELS.add_component
class NeuS(nn.Layer):
    def __init__(self,
                 coarse_ray_sampler: BaseSampler,
                 fine_ray_sampler: BaseSampler,
                 inside_field: BaseDensityField,
                 global_field: BaseDensityField,
                 rgb_renderer: nn.Layer,
                 outside_ray_sampler: Optional[BaseSampler] = None,
                 fine_sampling_steps: int = 4,
                 loss_weight_color: float = 1.0,
                 loss_weight_idr: float = 1.0,
                 loss_weight_mask: float = 1.0,
                 anneal_end: float = 0.0):
        super(NeuS, self).__init__()
        self.coarse_ray_sampler = coarse_ray_sampler
        self.fine_ray_sampler = fine_ray_sampler
        self.outside_ray_sampler = outside_ray_sampler
        self.inside_field = inside_field
        self.global_field = global_field
        self.rgb_renderer = rgb_renderer
        self.fine_sampling_steps = fine_sampling_steps
        self.loss_weight_color = loss_weight_color
        self.loss_weight_idr = loss_weight_idr
        self.loss_weight_mask = loss_weight_mask
        self.anneal_end = anneal_end

        # num_importance means resampling num. for inside_field.
        if fine_ray_sampler is None or fine_sampling_steps == 0:
            self.num_importance = 0
        else:
            assert fine_ray_sampler is not None
            assert fine_sampling_steps > 0
            self.num_importance = fine_sampling_steps * fine_ray_sampler.num_samples

    def _forward(self, sample: Tuple[RayBundle, dict],
                 cur_iter: int = None) -> Dict[str, paddle.Tensor]:
        if self.training:
            ray_bundle, pixel_batch = sample
        else:
            ray_bundle = sample

        # Sampling inside
        ray_samples_inside = self.coarse_ray_sampler(ray_bundle)
        num_inside = self.coarse_ray_sampler.num_samples

        if self.num_importance > 0:
            num_inside = num_inside + self.num_importance
            with paddle.no_grad():
                signed_distances = self.inside_field.sdf_fn(
                    ray_samples_inside.frustums.bin_points)

                for i in range(self.fine_sampling_steps):
                    batch_size = ray_samples_inside.spacing_bins.shape[0]
                    cos_val = get_cosine_coeff(
                        ray_samples_inside, signed_distances, radius_filter=1.0)
                    # Use middle sdf
                    if signed_distances.ndim == 3:
                        prev_sdf = signed_distances[:, :-1, :]
                        next_sdf = signed_distances[:, 1:, :]
                    else:
                        prev_sdf = signed_distances[::2, :]
                        next_sdf = signed_distances[1::2, :]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5  # [b, n_samples, 1]

                    inside_alpha = render_alpha_from_sdf(
                        ray_samples_inside,
                        mid_sdf,
                        inv_s=64 * 2**i,
                        coeff=cos_val,
                        clip_alpha=False)
                    inside_weights = render_weights_from_alpha(inside_alpha)
                    ray_samples_inside_new = self.fine_ray_sampler(
                        ray_bundle=ray_bundle,
                        ray_samples=ray_samples_inside,
                        weights=inside_weights)
                    # Merge inside samples and signed_distances
                    ray_samples_inside, index = ray_samples_inside.merge_samples(
                        [ray_samples_inside_new], ray_bundle, mode="sort")

                    if i + 1 != self.fine_sampling_steps:
                        new_signed_distances = self.inside_field.sdf_fn(
                            ray_samples_inside_new.frustums.bin_points)

                        signed_distances = paddle.concat(
                            [signed_distances, new_signed_distances], axis=-2)
                        num_inside_for_step_i = ray_samples_inside.frustums.bin_points.shape[
                            1]

                        batch_id = paddle.expand(
                            paddle.arange(batch_size)[:, None],
                            shape=[batch_size, num_inside_for_step_i])

                        batch_id = batch_id.reshape([-1])
                        index = index.reshape([-1])
                        signed_distances = signed_distances[(
                            batch_id, index)].reshape(
                                [batch_size, num_inside_for_step_i, 1])

        # Get inside field (SDF + RenderNet) outputs.
        inside_outputs = self.inside_field(ray_samples_inside)

        anneal_coeff = get_anneal_coeff(
            ray_samples_inside, inside_outputs["gradients"], cur_iter,
            self.anneal_end if self.training else 0.0)  # cos_anneal

        inside_alphas = render_alpha_from_sdf(
            ray_samples_inside, inside_outputs["sdf"], self.inside_field.inv_s,
            anneal_coeff)

        # Filtering points by their norm
        inside_sphere, relax_inside_sphere = self.inside_field.get_inside_filter_by_norm(
            ray_samples_inside)

        # Get the final alpha and color.
        if self.outside_ray_sampler is not None:
            ray_samples_outside = self.outside_ray_sampler(
                ray_bundle)  # Sampling outside
            global_ray_samples = ray_samples_inside.merge_samples(
                [ray_samples_outside], mode="force_concat")

            global_outputs = self.global_field(global_ray_samples)
            global_alphas = render_alpha_from_densities(
                ray_samples=global_ray_samples,
                densities=global_outputs["density"])

            alphas = inside_alphas * inside_sphere + global_alphas[
                ..., :num_inside, :] * (1.0 - inside_sphere)
            alphas = paddle.concat([alphas, global_alphas[..., num_inside:, :]],
                                   axis=1)
            colors = inside_outputs["rgb"] * inside_sphere + global_outputs[
                "rgb"][..., :num_inside, :] * (1.0 - inside_sphere)
            colors = paddle.concat(
                [colors, global_outputs["rgb"][:, num_inside:, :]], axis=1)
        else:
            alphas = inside_alphas
            colors = inside_outputs["rgb"]

        # Compute weights from alpha
        alphas = alphas.squeeze(-1)
        weights = alphas * paddle.cumprod(
            paddle.concat([paddle.ones([batch_size, 1]), 1. - alphas + 1e-7],
                          -1), -1)[:, :-1]
        weights = weights.unsqueeze(axis=-1)

        rgbs, visibility_mask = self.rgb_renderer(
            colors, weights, return_visibility=self.training)

        weights_sum = weights.squeeze(axis=-1).sum(axis=-1, keepdim=True)

        outputs = dict(rgb=rgbs)

        if self.training:
            # Eikonal loss
            gradient_error = (paddle.linalg.norm(
                inside_outputs["gradients"], p=2, axis=-1) - 1.0)**2
            gradient_error = gradient_error.unsqueeze(-1)

            eikonal_loss = (relax_inside_sphere * gradient_error).sum() / (
                relax_inside_sphere.sum() + 1e-5)

            # rgb loss
            rgb_loss = F.l1_loss(rgbs[visibility_mask],
                                 pixel_batch["pixels"][visibility_mask])

            # Mask loss (in progress)
            ones = paddle.ones_like(weights_sum)
            mask_loss = F.binary_cross_entropy(
                weights_sum.clip(1e-3, 1.0 - 1e-3), ones)

            outputs["loss"] = dict(
                rgb_loss=self.loss_weight_color * rgb_loss,
                eikonal_loss=self.loss_weight_idr * eikonal_loss,
                mask_loss=self.loss_weight_mask * mask_loss)

        num_samples_all = num_inside

        if self.outside_ray_sampler is not None:
            num_samples_all += self.outside_ray_sampler.num_samples

        outputs["num_samples_per_batch"] = num_samples_all
        normals = inside_outputs[
            "gradients"] * weights[:, :num_inside, :] * inside_sphere

        normals = paddle.sum(normals, axis=1)
        outputs["normals"] = normals

        return outputs

    def forward(self, *args, **kwargs):
        if hasattr(self, "amp_cfg_") and self.training:
            with paddle.amp.auto_cast(**self.amp_cfg_):
                return self._forward(*args, **kwargs)
        return self._forward(*args, **kwargs)

    def extract_fields(self, bound_min, bound_max, resolution):
        N = 64
        X = paddle.linspace(bound_min[0], bound_max[0], resolution).split(
            int(resolution / N))
        Y = paddle.linspace(bound_min[1], bound_max[1], resolution).split(
            int(resolution / N))
        Z = paddle.linspace(bound_min[2], bound_max[2], resolution).split(
            int(resolution / N))

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with paddle.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = paddle.meshgrid(xs, ys, zs)
                        pts = paddle.concat([
                            xx.reshape((-1, 1)),
                            yy.reshape((-1, 1)),
                            zz.reshape((-1, 1))
                        ],
                                            axis=-1)
                        pts = pts.unsqueeze(axis=0)
                        val = -self.inside_field.get_sdf_from_pts(pts)
                        val = val.reshape((len(xs), len(ys), len(zs))).numpy()
                        u[xi * N:xi * N + len(xs), yi * N:yi * N +
                          len(ys), zi * N:zi * N + len(zs)] = val
        return u

    def extract_geometry(self, bound_min, bound_max, resolution, threshold):
        logger.info('threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max
        b_min_np = bound_min
        vertices = vertices / (resolution - 1.0) * (
            b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        return vertices, triangles
