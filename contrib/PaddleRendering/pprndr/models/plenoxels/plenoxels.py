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
from pprndr.models.ray_samplers import GridIntersectionSampler
from pprndr.ray_marching import render_weight_from_density

__all__ = ["Plenoxels"]


@manager.MODELS.add_component
class Plenoxels(nn.Layer):
    def __init__(self,
                 ray_sampler: GridIntersectionSampler,
                 field: BaseDensityField,
                 rgb_renderer: nn.Layer,
                 rgb_loss: nn.Layer = nn.MSELoss()):
        super(Plenoxels, self).__init__()

        assert isinstance(ray_sampler, GridIntersectionSampler), \
            "Plenoxels currently only supports GridIntersectionSampler."
        self.ray_sampler = ray_sampler
        self.field = field
        self.rgb_renderer = rgb_renderer
        self.rgb_loss = rgb_loss

    def _forward(self,
                 sample: Union[Tuple[RayBundle, dict], RayBundle],
                 cur_iter: int = None) -> Dict[str, paddle.Tensor]:
        if self.training:
            ray_bundle, pixel_batch = sample
        else:
            ray_bundle = sample

        with paddle.no_grad():
            ray_samples = self.ray_sampler(ray_bundle, plenoxel_grid=self.field)

        field_outputs = self.field(ray_samples)

        weights = render_weight_from_density(
            t_starts=ray_samples.frustums.starts,
            t_ends=ray_samples.frustums.ends,
            densities=field_outputs["density"],
            packed_info=ray_samples.packed_info)

        accumulated_rgb, _ = self.rgb_renderer(
            field_outputs["rgb"],
            weights,
            ray_indices=ray_samples.ray_indices,
            num_rays=ray_bundle.num_rays,
            return_visibility=False)

        outputs = dict(rgb=accumulated_rgb)

        if self.training:
            rgb_loss = self.rgb_loss(accumulated_rgb, pixel_batch["pixels"])
            outputs["num_samples_per_batch"] = ray_samples.packed_info[-1].sum()
            outputs["loss"] = dict(rgb_loss=rgb_loss)

        return outputs

    def forward(self, *args, **kwargs):
        if hasattr(self, "amp_cfg_") and self.training:
            with paddle.amp.auto_cast(**self.amp_cfg_):
                return self._forward(*args, **kwargs)
        return self._forward(*args, **kwargs)
