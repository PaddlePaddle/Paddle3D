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

import math
from dataclasses import dataclass

import paddle

__all__ = ["conical_frustum_to_gaussian", "Gaussians"]


@dataclass
class Gaussians:
    mean: paddle.Tensor
    covariance: paddle.Tensor


def conical_frustum_to_gaussian(
        origins: paddle.Tensor, directions: paddle.Tensor,
        pixel_area: paddle.Tensor, starts: paddle.Tensor,
        ends: paddle.Tensor) -> Gaussians:
    """
    Approximates conical frustums with a Gaussian distributions.
    """
    cone_radius = paddle.sqrt(pixel_area / math.pi)
    mu = (starts + ends) / 2.0
    hw = (ends - starts) / 2.0
    means = origins + directions * (mu + (2.0 * mu * hw**2.0) /
                                    (3.0 * mu**2.0 + hw**2.0))
    dir_variance = (hw**2) / 3 - (4 / 15) * (
        (hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2)**2)
    radius_variance = cone_radius**2 * (
        (mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2))
    return compute_3d_gaussian(directions, means, dir_variance, radius_variance)


def compute_3d_gaussian(directions: paddle.Tensor, means: paddle.Tensor,
                        dir_variance: paddle.Tensor,
                        radius_variance: paddle.Tensor) -> Gaussians:
    dir_outer_product = directions[..., :, None] * directions[..., None, :]
    eye = paddle.eye(directions.shape[-1])
    dir_mag_sq = paddle.clip(
        paddle.sum(directions**2, axis=-1, keepdim=True), min=1e-10)
    null_outer_product = eye - directions[..., :, None] * (
        directions / dir_mag_sq)[..., None, :]
    dir_cov_diag = dir_variance[..., None] * dir_outer_product[..., :, :]
    radius_cov_diag = radius_variance[..., None] * null_outer_product[..., :, :]
    covariance = dir_cov_diag + radius_cov_diag

    return Gaussians(mean=means, covariance=covariance)
