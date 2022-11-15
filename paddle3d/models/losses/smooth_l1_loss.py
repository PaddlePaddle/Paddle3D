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

import paddle


def smooth_l1_loss(input, target, beta, reduction="none"):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:

                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.

    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

    Returns:
        The loss with the reduction option applied.
    This code is based on https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
     """
    beta = paddle.to_tensor(beta).cast(input.dtype)
    if beta < 1e-5:
        loss = paddle.abs(input - target)
    else:
        n = paddle.abs(input - target)
        loss = paddle.where(n < beta, 0.5 * n ** 2, n - 0.5 * beta)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
