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

from typing import Tuple, Union

import paddle
import paddle.nn as nn

from pprndr.apis import manager
from pprndr.cameras.rays import RaySamples
from pprndr.ray_marching import accumulate_along_rays
from pprndr.utils.colors import get_color

__all__ = ["RGBRenderer", "AccumulationRenderer", "DepthRenderer"]


@manager.RENDERERS.add_component
class RGBRenderer(nn.Layer):
    def __init__(self, background_color: Union[str, list, tuple] = "random"):
        super(RGBRenderer, self).__init__()

        if isinstance(background_color, str):
            if background_color not in ["random", "last_sample"]:
                self.background_color = paddle.to_tensor(
                    get_color(background_color))
            else:
                self.background_color = background_color
        else:
            self.background_color = paddle.to_tensor(background_color)

    def forward(
            self,
            rgb: paddle.Tensor,
            weights: paddle.Tensor,
            ray_indices: paddle.Tensor = None,
            num_rays: int = None,
            return_visibility: bool = False
    ) -> Union[Tuple[paddle.Tensor, None], Tuple[paddle.Tensor, paddle.Tensor]]:
        """
        Args:
            rgb: [N_rays, N_samples, 3]
            weights: [N_rays, N_samples]
            ray_indices: [N_samples]
            num_rays: int
            return_visibility: bool
        """

        if ray_indices is not None and num_rays is not None:
            accumulated_rgb = accumulate_along_rays(weights * rgb, ray_indices,
                                                    num_rays)
            accumulated_weight = accumulate_along_rays(weights, ray_indices,
                                                       num_rays)
        else:
            accumulated_rgb = paddle.sum(weights * rgb, axis=-2)
            accumulated_weight = paddle.sum(weights, axis=-2)

        if isinstance(self.background_color, paddle.Tensor):
            background_color = self.background_color
        elif self.background_color == "random":
            background_color = paddle.rand(accumulated_rgb.shape)
        elif self.background_color == "last_sample":
            background_color = rgb[..., -1, :]

        # blend with background color
        accumulated_rgb += (1.0 - accumulated_weight) * background_color

        if not self.training:
            accumulated_rgb = paddle.clip(accumulated_rgb, 0.0, 1.0)

        if return_visibility:
            return accumulated_rgb, accumulated_weight.squeeze(-1) > 0.
        else:
            return accumulated_rgb, None


@manager.RENDERERS.add_component
class AccumulationRenderer(nn.Layer):
    def forward(self,
                weights: paddle.Tensor,
                ray_indices: paddle.Tensor = None,
                num_rays: int = None):
        """
        Args:
            weights: [N_rays, N_samples]
            ray_indices: [N_samples]
            num_rays: int
        """

        if ray_indices is not None and num_rays is not None:
            accumulated_weight = accumulate_along_rays(weights, ray_indices,
                                                       num_rays)
        else:
            accumulated_weight = paddle.sum(weights, axis=-2)

        return accumulated_weight


@manager.RENDERERS.add_component
class DepthRenderer(nn.Layer):
    """Calculate depth along ray.

        Depth Method:
            - median: Depth is set to the distance where the accumulated weight reaches 0.5.
            - expected: Expected depth along ray. Same procedure as rendering rgb, but with depth.

        Args:
            method: Depth calculation method.
    """

    def __init__(self, method: str = "median"):
        super(DepthRenderer, self).__init__()

        assert method in ["median", "expected"
                          ], "Depth method should be 'median' or 'expected'."

        self.method = method

    def forward(self,
                weights: paddle.Tensor,
                ray_samples: RaySamples,
                ray_indices: paddle.Tensor = None,
                num_rays: int = None):
        """
        Composite samples along ray and calculate depths.
        """

        if self.method == "median":
            steps = (
                ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
            if ray_indices is not None and num_rays is not None:
                raise NotImplementedError(
                    "Median depth calculation is not implemented for packed samples."
                )
            cumulative_weights = paddle.cumsum(
                weights[..., 0], axis=-1)  # [..., num_samples]
            split = paddle.full((*weights.shape[:-2], 1), .5)  # [..., 1]
            median_index = paddle.searchsorted(cumulative_weights,
                                               split)  # [..., 1]
            median_index = paddle.clip(median_index, 0,
                                       steps.shape[-2] - 1)  # [..., 1]
            median_depth = paddle.take_along_axis(
                steps[..., 0], axis=-1, indices=median_index)  # [..., 1]
            return median_depth

        elif self.method == "expected":
            eps = 1e-10
            steps = (
                ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

            if ray_indices is not None and num_rays is not None:
                depth = accumulate_along_rays(weights * steps, ray_indices,
                                              num_rays)
                accumulation = accumulate_along_rays(weights, ray_indices,
                                                     num_rays)
                depth = depth / (accumulation + eps)
            else:
                depth = paddle.sum(
                    weights * steps, axis=-2) / (paddle.sum(weights, -2) + eps)

            depth = paddle.clip(depth, steps.min(), steps.max())

            return depth
