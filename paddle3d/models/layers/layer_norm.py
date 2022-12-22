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
import paddle.nn as nn
import paddle.nn.functional as F


def group_norm(out_channels):
    """group normal function

    Args:
        out_channels (int): out channel nums

    Returns:
        nn.Layer: GroupNorm op
    """
    num_groups = 32
    if out_channels % 32 == 0:
        return nn.GroupNorm(num_groups, out_channels)
    else:
        return nn.GroupNorm(num_groups // 2, out_channels)


class FrozenBatchNorm2d(nn.Layer):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.

    This code is based on https://github.com/facebookresearch/detectron2/blob/32b61e64c76118b2e9fc2237f283a8e9c938bd16/detectron2/layers/batch_norm.py#L13
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", paddle.ones([num_features]))
        self.register_buffer("bias", paddle.zeros([num_features]))
        self.register_buffer("_mean", paddle.zeros([num_features]))
        self.register_buffer("_variance", paddle.ones([num_features]) - eps)

    def forward(self, x):
        if not x.stop_gradient:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self._variance + self.eps).rsqrt()
            bias = self.bias - self._mean * scale
            scale = scale.reshape([1, -1, 1, 1])
            bias = bias.reshape([1, -1, 1, 1])
            out_dtype = x.dtype  # may be half
            return x * scale.cast(out_dtype) + bias.cast(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self._mean,
                self._variance,
                self.weight,
                self.bias,
                training=False,
                epsilon=self.eps,
            )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(
            self.num_features, self.eps)
