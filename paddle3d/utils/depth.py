# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/depth_utils.py#L4

    Converts depth map into bin indices
    Args:
        depth_map: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * paddle.sqrt(1 + 8 *
                                           (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (paddle.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indices outside of bounds
        mask = (indices < 0) | (indices >
                                num_bins) | (~paddle.isfinite(indices))
        indices[mask] = num_bins
        sub_val = paddle.full(shape=mask.shape, fill_value=num_bins)
        indices_ = paddle.where(mask, sub_val, indices)

        # Convert to integer
        indices = indices_.cast("int64")
    return indices
