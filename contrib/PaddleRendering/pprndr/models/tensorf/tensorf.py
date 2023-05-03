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

from typing import Dict, Tuple, Union, Generic, Any
from collections.abc import Iterable, Mapping
import numpy as np

import paddle
import paddle.nn as nn

from pprndr.apis import manager
from pprndr.cameras.rays import RayBundle, RaySamples
from pprndr.models.fields import TensoRFField
from pprndr.models.ray_samplers import BaseSampler
from pprndr.ray_marching import render_weight_from_density

__all__ = ["TensoRF"]


@manager.MODELS.add_component
class TensoRF(nn.Layer):
    def __init__(self,
                 coarse_ray_sampler: BaseSampler,
                 fine_ray_sampler: BaseSampler,
                 field: TensoRFField,
                 rgb_renderer: nn.Layer,
                 accumulation_renderer: nn.Layer,
                 fine_rgb_loss: nn.Layer = nn.MSELoss(),
                 init_resolution: int = 128,
                 final_resolution: int = 300,
                 upsampling_iters: Tuple[int, ...] = (2000, 3000, 4000, 5500,
                                                      7000),
                 L1_weight_inital: float = 8.0e-5):
        super(TensoRF, self).__init__()

        self.coarse_ray_sampler = coarse_ray_sampler
        self.fine_ray_sampler = fine_ray_sampler
        self.field = field
        self.rgb_renderer = rgb_renderer
        self.accumulation_renderer = accumulation_renderer
        self.fine_rgb_loss = fine_rgb_loss

        self.L1_weight_inital = L1_weight_inital

        # self.update_AlphaMask_iters = update_AlphaMask_iters # 为光线设置mask
        self.upsampling_iters = upsampling_iters
        self.upsampling_steps = (np.round(
            np.exp(
                np.linspace(
                    np.log(init_resolution),
                    np.log(final_resolution),
                    len(upsampling_iters) + 1,
                ))).astype("int").tolist()[1:])

    def reinitialize_optimizer(
            self, reinit_optim_cfg: dict,
            gard_vars: Union[list, tuple]) -> paddle.optimizer.Optimizer:
        lr_params = reinit_optim_cfg.get('lr_scheduler')
        lr_scheduler = self._load_object(lr_params)

        params = reinit_optim_cfg.get('optimizer', {}).copy()
        params['learning_rate'] = lr_scheduler
        params['parameters'] = gard_vars

        return self._load_object(params)

    def _load_component(self, com_name: str) -> Any:
        # lazy import
        import pprndr.apis.manager as manager

        for com in manager.__all__:
            com = getattr(manager, com)
            if com_name in com.components_dict:
                return com[com_name]
        else:
            if com_name in paddle.optimizer.lr.__all__:
                return getattr(paddle.optimizer.lr, com_name)
            elif com_name in paddle.optimizer.__all__:
                return getattr(paddle.optimizer, com_name)

            raise RuntimeError(
                'The specified component was not found {}.'.format(com_name))

    def _load_object(self, obj: Generic, recursive: bool = True) -> Any:
        if isinstance(obj, Mapping):
            dic = obj.copy()
            component = self._load_component(
                dic.pop('type')) if 'type' in dic else dict

            if recursive:
                params = {}
                for key, val in dic.items():
                    params[key] = self._load_object(
                        obj=val, recursive=recursive)
            else:
                params = dic

            return component(**params)

        elif isinstance(obj, Iterable) and not isinstance(obj, str):
            return [self._load_object(item) for item in obj]

        return obj

    def update_to_step(self, step: int) -> None:
        index = self.upsampling_iters.index(step)
        new_grid_resolution = self.upsampling_steps[index]

        self.field.density_encoder.upsample_grid(
            new_grid_resolution)  # type: ignore
        self.field.color_encoder.upsample_grid(
            new_grid_resolution)  # type: ignore
        self.coarse_ray_sampler.occupancy_grid.upsample(new_grid_resolution)

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

        # pdf sampling
        fine_samples = self.fine_ray_sampler(
            ray_bundle=ray_bundle,
            ray_samples=coarse_samples,
            weights=coarse_weights)

        num_fine_samples = self.fine_ray_sampler.num_samples
        if getattr(self.fine_ray_sampler, "include_original", False):
            num_coarse_samples = self.coarse_ray_sampler.num_samples
            num_fine_samples = num_fine_samples + num_coarse_samples + 1

        # fine field:
        field_outputs_fine = self.field(fine_samples)

        fine_weights = render_weight_from_density(
            t_starts=fine_samples.frustums.starts,
            t_ends=fine_samples.frustums.ends,
            densities=field_outputs_fine["density"],
            packed_info=fine_samples.packed_info)

        fine_rgb, fine_visibility_mask = self.rgb_renderer(
            field_outputs_fine["rgb"],
            fine_weights,
            ray_indices=fine_samples.ray_indices,
            num_rays=ray_bundle.num_rays,
            return_visibility=self.training)

        outputs = dict(rgb=fine_rgb)

        if self.training:
            L1_reg_weight = self.L1_weight_inital
            fine_rgb_loss = self.fine_rgb_loss(
                fine_rgb[fine_visibility_mask],
                pixel_batch["pixels"][fine_visibility_mask])
            reg_L1_loss = self.field.density_L1()
            if L1_reg_weight != 0:
                outputs["loss"] = dict(
                    fine_rgb_loss=fine_rgb_loss,
                    reg_L1_loss=L1_reg_weight * reg_L1_loss)
            else:
                outputs["loss"] = dict(fine_rgb_loss=fine_rgb_loss)
            outputs["num_samples_per_batch"] = num_fine_samples

        return outputs

    def forward(self, *args, **kwargs):
        if hasattr(self, "amp_cfg_") and self.training:
            with paddle.amp.auto_cast(**self.amp_cfg_):
                return self._forward(*args, **kwargs)
        return self._forward(*args, **kwargs)
