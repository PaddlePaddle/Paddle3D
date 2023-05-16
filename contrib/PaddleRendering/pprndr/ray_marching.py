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

from typing import Any, Callable, Tuple

import numpy as np
import paddle
from paddle.nn import functional as F

try:
    import ray_marching_lib
except ModuleNotFoundError:
    from pprndr.cpp_extensions import ray_marching_lib

from pprndr.cameras.rays import RaySamples
from pprndr.geometries import ContractionType

__all__ = [
    "near_far_from_aabb", "near_far_from_sphere", "grid_query", "contract",
    "contract_inv", "transmittance_from_alpha", "unpack_info", "ray_marching",
    "pdf_ray_marching", "render_visibility", "pack_info",
    "render_weight_from_density", "accumulate_along_rays",
    "render_alpha_from_densities", "render_alpha_from_sdf",
    "render_weights_from_alpha", "render_alpha_from_sdf", "get_anneal_coeff",
    "get_cosine_coeff"
]


@paddle.no_grad()
def near_far_from_sphere(rays_o, rays_d):
    a = paddle.sum(rays_d**2, axis=-1, keepdim=True)
    b = 2.0 * paddle.sum(rays_o * rays_d, axis=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    return near, far


@paddle.no_grad()
def near_far_from_aabb(rays_origins: paddle.Tensor,
                       rays_directions: paddle.Tensor, aabb: paddle.Tensor):
    nears, fars = ray_marching_lib.near_far_from_aabb(rays_origins,
                                                      rays_directions, aabb)

    return nears, fars


@paddle.no_grad()
def ray_marching_op(origins: paddle.Tensor, directions: paddle.Tensor,
                    nears: paddle.Tensor, fars: paddle.Tensor,
                    grid_aabb: paddle.Tensor, grid_binary: paddle.Tensor,
                    contraction_type: ContractionType, step_size: float,
                    cone_angle: float):
    # count number of samples per ray
    num_steps = ray_marching_lib.first_round_ray_marching(
        origins, directions, nears, fars, grid_aabb, grid_binary,
        contraction_type.value, step_size, cone_angle)
    cum_steps = paddle.cumsum(num_steps, axis=0)
    packed_info = paddle.stack([cum_steps - num_steps, num_steps], axis=1)
    total_steps = cum_steps[-1].item()

    if total_steps == 0:
        packed_info[-1, 1] = 1
        starts = paddle.ones([1, 1])
        ends = paddle.ones([1, 1])
        return packed_info, starts, ends, total_steps

    # output samples starts and ends
    starts, ends = ray_marching_lib.second_round_ray_marching(
        origins, directions, nears, fars, grid_aabb, grid_binary, packed_info,
        contraction_type.value, step_size, cone_angle, total_steps)

    return packed_info, starts, ends, total_steps


@paddle.no_grad()
def grid_query(samples: paddle.Tensor, aabb: paddle.Tensor,
               grid_values: paddle.Tensor,
               contraction_type: ContractionType) -> paddle.Tensor:
    return ray_marching_lib.grid_query(samples, aabb, grid_values,
                                       contraction_type.value)


@paddle.no_grad()
def contract(samples: paddle.Tensor, aabb: paddle.Tensor,
             contraction_type: ContractionType) -> paddle.Tensor:
    return ray_marching_lib.contract(samples, aabb, contraction_type.value)


@paddle.no_grad()
def contract_inv(samples: paddle.Tensor, aabb: paddle.Tensor,
                 contraction_type: ContractionType) -> paddle.Tensor:
    return ray_marching_lib.contract_inv(samples, aabb, contraction_type.value)


def transmittance_from_alpha(packed_info: paddle.Tensor,
                             alphas: paddle.Tensor) -> paddle.Tensor:
    return ray_marching_lib.transmittance_from_alpha(packed_info, alphas)


@paddle.no_grad()
def unpack_info(packed_info: paddle.Tensor) -> paddle.Tensor:
    n_samples = packed_info[-1].sum(0).item()
    ray_indices = ray_marching_lib.unpack_info(packed_info, n_samples)
    return ray_indices


@paddle.no_grad()
def ray_marching(origins: paddle.Tensor,
                 directions: paddle.Tensor,
                 nears: paddle.Tensor = None,
                 fars: paddle.tensor = None,
                 scene_aabb: paddle.Tensor = None,
                 compute_near_far_from_sphere: bool = False,
                 occupancy_grid: Any = None,
                 sigma_fn: Callable = None,
                 alpha_fn: Callable = None,
                 transmittance_thresh: float = 1e-4,
                 alpha_thresh: float = 1e-2,
                 min_near: float = None,
                 max_far: float = None,
                 step_size: float = 1e-3,
                 stratified: bool = False,
                 cone_angle: float = 0.0
                 ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    Ray marching.

    Args:
        origins (paddle.Tensor): Rays' origins of shape [N_rays, 3].
        directions (paddle.Tensor): Rays' directions of shape [N_rays, 3].
        nears (paddle.Tensor, optional): Per-ray minimum distance of marching beginning. Defaults to None.
        fars (paddle.Tensor, optional): Per-ray maximum distance of marching ending. Defaults to None.
        scene_aabb (paddle.Tensor, optional): AABB bound of the scene of shape [6], in the format of
            {x_min, y_min, z_min, x_max, y_max, z_max}. Defaults to None.
        occupancy_grid (Union[OccupancyGrid, PlenoxelGrid], optional): Occupancy grid of the scene. Defaults to None.
        sigma_fn (Callable, optional): Sigma (density) function. If provided, marching will skip invisible
            positions by evaluating density value along rays. Defaults to None. Only one of `sigma_fn` and
            `alpha_fn` can be provided.
        alpha_fn (Callable, optional): Alpha (opacity) function. If provided, marching will skip invisible
            positions by evaluating opacity value along rays. Defaults to None. Only one of `sigma_fn` and
            `alpha_fn` can be provided.
        transmittance_thresh (float, optional): Transmittance threshold for early stopping. Defaults to 1e-4.
        alpha_thresh (float, optional): Alpha threshold for early stopping. Defaults to 1e-2.
        min_near (float, optional): Minimum distance of marching beginning for all rays. Defaults to 0.0.
        max_far (float, optional): Maximum distance of marching ending for all rays. Defaults to 1e10.
        step_size (float, optional): Render step size. Defaults to 1e-3.
        stratified (bool, optional): Whether to use stratified sampling. Defaults to False.
        cone_angle (float, optional): Cone angle for linearly-increased step size. 0. means
            constant step size. Defaults to 0.0.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]: (packed_info, starts, ends)
    """

    if sigma_fn is not None and alpha_fn is not None:
        raise ValueError(
            "Only one of `sigma_fn` and `alpha_fn` can be provided.")

    if nears is None or fars is None:
        if scene_aabb is not None:
            nears, fars = near_far_from_aabb(origins, directions, scene_aabb)
        elif compute_near_far_from_sphere is not None:
            assert (scene_aabb is None)
            nears, fars = near_far_from_sphere(origins, directions, scene_aabb)
        else:
            nears = paddle.zeros_like(origins[:, 0]).unsqueeze_(-1)
            fars = paddle.full_like(origins[:, 0], 1e10).unsqueeze_(-1)

    if min_near is not None:
        nears = paddle.clip(nears, min=min_near)
    if max_far is not None:
        fars = paddle.clip(fars, max=max_far)

    if stratified:
        # stratified sampling
        nears += paddle.rand(nears.shape, dtype=nears.dtype) * step_size

    if occupancy_grid is not None:
        grid_aabb = occupancy_grid.aabb
        grid_binary = occupancy_grid.binary
        contraction_type = occupancy_grid.contraction_type
    else:
        grid_aabb = paddle.to_tensor([-1e10, -1e10, -1e10, 1e10, 1e10, 1e10])
        grid_binary = paddle.ones([1, 1, 1], dtype="bool")
        contraction_type = ContractionType.AABB

    packed_info, starts, ends, total_steps = ray_marching_op(
        origins,
        directions,
        nears,
        fars,
        grid_aabb=grid_aabb,
        grid_binary=grid_binary,
        contraction_type=contraction_type,
        step_size=step_size,
        cone_angle=cone_angle)

    if total_steps == 0:
        return packed_info, starts, ends

    if sigma_fn is not None or alpha_fn is not None:
        ray_indices = unpack_info(packed_info)
        if sigma_fn is not None:
            sigmas = sigma_fn(starts, ends, ray_indices)
            alphas = paddle.ones_like(sigmas) - paddle.exp(sigmas * starts -
                                                           sigmas * ends)
        else:
            alphas = alpha_fn(starts, ends, ray_indices)

        visibility_mask, packed_info_visible = render_visibility(
            packed_info, alphas, transmittance_thresh, alpha_thresh)
        if paddle.any(visibility_mask):
            starts = paddle.masked_select(starts.squeeze(-1),
                                          visibility_mask).unsqueeze(-1)
            ends = paddle.masked_select(ends.squeeze(-1),
                                        visibility_mask).unsqueeze(-1)
        else:
            starts = paddle.ones([1, 1], dtype=starts.dtype)
            ends = paddle.ones([1, 1], dtype=ends.dtype)
            packed_info_visible[-1, 1] = 1
        packed_info = packed_info_visible

    return packed_info, starts, ends


@paddle.no_grad()
def pdf_ray_marching(
        starts: paddle.Tensor, ends: paddle.Tensor, weights: paddle.Tensor,
        packed_info: paddle.Tensor,
        num_samples: int) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    if packed_info.sum().item() == 1:
        return packed_info, starts, ends

    resample_num_steps = paddle.where(
        packed_info[:, 1] > 0, paddle.to_tensor([num_samples], dtype="int32"),
        paddle.zeros([1], dtype="int32"))
    resample_cum_steps = paddle.cumsum(resample_num_steps, axis=0)
    resample_packed_info = paddle.stack(
        [resample_cum_steps - resample_num_steps, resample_num_steps], axis=1)
    resample_total_steps = resample_cum_steps[-1].item()

    resample_starts, resample_ends = ray_marching_lib.pdf_ray_marching(
        starts, ends, weights, packed_info, resample_packed_info,
        resample_total_steps)

    return resample_packed_info, resample_starts, resample_ends


@paddle.no_grad()
def render_visibility(
        packed_info: paddle.Tensor,
        alphas: paddle.Tensor,
        transmittance_thresh: float = 1e-4,
        alpha_thresh: float = 0.0) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Filter out invisible (transparent and occluded) samples.
    """
    num_steps, visibility_mask = ray_marching_lib.rendering_alphas_compressed(
        packed_info, alphas, transmittance_thresh, alpha_thresh)
    cum_steps = paddle.cumsum(num_steps, axis=0)
    packed_info_visible = paddle.stack([cum_steps - num_steps, num_steps],
                                       axis=1)

    return visibility_mask, packed_info_visible


@paddle.no_grad()
def pack_info(ray_indices: paddle.Tensor, n_rays: int = None) -> paddle.Tensor:
    if n_rays is None:
        n_rays = ray_indices.max() + 1

    updates = paddle.ones_like(ray_indices)
    num_steps = paddle.zeros((n_rays, ), dtype="int32")
    num_steps = paddle.put_along_axis(
        num_steps, ray_indices, updates, axis=0, reduce="add")
    cum_steps = paddle.cumsum(num_steps, axis=0)
    packed_info = paddle.stack([cum_steps - num_steps, num_steps], axis=-1)

    return packed_info


def render_weight_from_density(
        t_starts: paddle.Tensor,
        t_ends: paddle.Tensor,
        densities: paddle.Tensor,
        *,
        packed_info: paddle.Tensor = None) -> paddle.Tensor:
    """
    Render samples' weight from their densities along each ray.

    Args:
        t_starts: Starting distances of each interval. Shape: [n_rays, n_samples, 1] or [n, 1]
        t_ends: Ending distances of each interval. Shape: [n_rays, n_samples, 1] or [n, 1]
        densities: Densities of each sample. Shape: [n_rays, n_samples, 1] or [n, 1]
        packed_info: Packed information of each ray (if samples are packed). Shape: [n_rays, 2].
            The first column is the starting index of each ray, and the second column is the number
            of samples of each ray. Set to None if the samples are of shape [n_rays, n_samples, 1].
            Default: None.
    """
    if packed_info is None:
        weights = ray_marching_lib.weight_from_sigma(t_starts, t_ends,
                                                     densities)
    else:
        weights = ray_marching_lib.weight_from_packed_sigma(
            packed_info, t_starts, t_ends, densities)

    return weights


def accumulate_along_rays(values: paddle.Tensor,
                          ray_indices: paddle.Tensor,
                          n_rays: int = None) -> paddle.Tensor:
    """
    Accumulate values along rays.

    Args:
        values: Values to be accumulated. 2-dimensional tensor of shape [n, c].
        ray_indices: Ray indices of each sample. 1-dimensional tensor of shape [n].
        n_rays: Number of rays. If not provided, it will be inferred from ray_indices (n_rays = ray_indices.max() + 1).
            Default: None.
    """

    if ray_indices.numel() == 0:
        assert n_rays is not None
        return paddle.zeros((n_rays, values.shape[-1]))

    outputs = paddle.geometric.segment_sum(values, ray_indices)

    if n_rays is not None and len(outputs) < n_rays:
        outputs = paddle.nn.functional.pad(outputs,
                                           [0, n_rays - len(outputs), 0, 0])

    return outputs


def get_anneal_coeff(ray_samples: RaySamples = None,
                     gradients: paddle.Tensor = None,
                     cur_iter: int = None,
                     anneal_end: float = 0.0) -> float:
    dirs = ray_samples.frustums.directions
    true_cos = (dirs * gradients).sum(-1, keepdim=True)
    if anneal_end == 0.0:
        cos_anneal_ratio = 1.0
    else:
        assert (cur_iter is not None)
        cos_anneal_ratio = np.min([1.0, cur_iter / anneal_end])

    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + \
                 F.relu(-true_cos) * cos_anneal_ratio)
    return iter_cos


def get_cosine_coeff(ray_samples: RaySamples = None,
                     signed_distance: paddle.Tensor = None,
                     radius_filter: float = None) -> paddle.Tensor:
    if signed_distance.ndim == 3:
        prev_sdf = signed_distance[:, :-1, :]  # [b, n_samples, 1]
        next_sdf = signed_distance[:, 1:, :]  # [b, n_samples, 1]
    else:
        prev_sdf = signed_distance[::2, :]  # [n_total_samples, 1]
        next_sdf = signed_distance[1::2, :]  # [n_total_samples, 1]

    prev_z_vals = ray_samples.frustums.starts  # [b, n_samples, 1] or [n_total_samples, 1]
    next_z_vals = ray_samples.frustums.ends  # [b, n_samples, 1] or [n_total_samples, 1]

    cos_val = (next_sdf - prev_sdf) / (
        next_z_vals - prev_z_vals + 1e-5
    )  # [b, n_samples, 1] or [n_total_samples, 1]
    prev_cos_val = paddle.concat(
        [paddle.zeros_like(cos_val[..., :1, :]), cos_val[..., :-1, :]],
        axis=-2)  # [b, n_samples, 1] or [n_total_samples, 1]
    cos_val = paddle.concat([prev_cos_val, cos_val],
                            axis=-1)  # [b, n_samples, 2]
    cos_val = paddle.min(cos_val, axis=-1, keepdim=True)  # [b, n_samples, 1]
    cos_val = cos_val.clip(-1e3, 0.0)

    if radius_filter is not None:
        inside_pts = ray_samples.frustums.bin_points
        radius = paddle.linalg.norm(inside_pts, p=2, axis=-1, keepdim=True)
        inside_sphere = (radius[..., :-1, :] < 1.0) | (radius[..., 1:, :] < 1.0)
        cos_val = cos_val * inside_sphere

    return cos_val


def render_alpha_from_sdf(ray_samples,
                          signed_distances,
                          inv_s,
                          coeff=1.0,
                          clip_alpha=True):
    # Note: coeff limits sdf interval
    dists = ray_samples.frustums.ends - ray_samples.frustums.starts
    estimated_next_sdf = signed_distances + coeff * dists * 0.5
    estimated_prev_sdf = signed_distances - coeff * dists * 0.5

    prev_cdf = F.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = F.sigmoid(estimated_next_sdf * inv_s)
    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5))
    if clip_alpha:
        alpha = alpha.clip(0.0, 1.0)

    return alpha


def render_alpha_from_densities(ray_samples: RaySamples = None,
                                densities: paddle.Tensor = None
                                ) -> paddle.Tensor:
    ends = ray_samples.frustums.ends
    starts = ray_samples.frustums.starts
    intervals = ends - starts
    directions = paddle.unsqueeze(ray_samples.frustums.directions, axis=-2)
    deltas = intervals * paddle.linalg.norm(directions, axis=-1)
    density_deltas = densities * deltas
    alpha = 1 - paddle.exp(-density_deltas)
    return alpha


def render_weights_from_alpha(alpha):
    batch_size = alpha.shape[0]
    weights = alpha * paddle.cumprod(
        paddle.concat([paddle.ones([batch_size, 1, 1]), 1. - alpha + 1e-7], 1),
        1)[:, :-1, :]
    return weights
