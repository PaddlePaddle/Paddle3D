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

from typing import Tuple

import numpy as np
import paddle

__all__ = ["radial_n_tangential_undistort", "get_distortion_coeffs"]


def radial_n_tangential_undistort(xy_coords: paddle.Tensor,
                                  distortion_coeffs: paddle.Tensor,
                                  eps: float = 1e-9,
                                  max_iterations: int = 10) -> paddle.Tensor:
    """
    Undistort camera coordinates using radial and tangential distortion
    coefficients.
    Adapted from MultiNeRF
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509


    Args:
        xy_coords (Tensor): XY coordinates to undistort, of shape (..., 2).
        distortion_coeffs (Tensor): Distortion coefficients [k1, k2, k3, k4, p1, p2].
        eps (float): Tolerance for convergence.
        max_iterations (int): Maximum number of iterations.

    Returns:
        Tensor: Undistorted camera coordinates.
    """

    x = xy_coords[..., 0]
    y = xy_coords[..., 1]

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x,
            y=y,
            xd=xy_coords[..., 0],
            yd=xy_coords[..., 1],
            distortion_coeffs=distortion_coeffs)

        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = paddle.where(
            paddle.abs(denominator) > eps, x_numerator / denominator,
            paddle.to_tensor(0.))
        step_y = paddle.where(
            paddle.abs(denominator) > eps, y_numerator / denominator,
            paddle.to_tensor(0.))

        x += step_x
        y += step_y

    return paddle.stack([x, y], axis=-1)


def _compute_residual_and_jacobian(
        x: paddle.Tensor, y: paddle.Tensor, xd: paddle.Tensor,
        yd: paddle.Tensor, distortion_coeffs: paddle.Tensor
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.
           Tensor, paddle.Tensor]:
    # TODO(will-jl944): This could be implemented as a cpp extension.
    """
    Auxiliary function of radial_n_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_coeffs: The distortion coefficients [k1, k2, k3, k4, p1, p2].

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
    """

    k1 = distortion_coeffs[..., 0]
    k2 = distortion_coeffs[..., 1]
    k3 = distortion_coeffs[..., 2]
    k4 = distortion_coeffs[..., 3]
    p1 = distortion_coeffs[..., 4]
    p2 = distortion_coeffs[..., 5]

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def get_distortion_coeffs(k1: float = 0.,
                          k2: float = 0.,
                          k3: float = 0.,
                          k4: float = 0.,
                          p1: float = 0.,
                          p2: float = 0.) -> np.ndarray:
    """
    Returns an matrix of distortion coefficients.

    Args:
        k1: 1st Radial distortion coefficient.
        k2: 2nd Radial distortion coefficient.
        k3: 3rd Radial distortion coefficient.
        k4: 4th Radial distortion coefficient.
        p1: 1st Tangential distortion coefficient.
        p2: 2nd Tangential distortion coefficient.

    Returns:
        np.ndarray: Distortion coefficients [k1, k2, k3, k4, p1, p2].
    """
    return np.array([k1, k2, k3, k4, p1, p2], dtype=np.float32)
