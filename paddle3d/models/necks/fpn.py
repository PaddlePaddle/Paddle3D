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
from paddle import nn
import paddle.nn.functional as F

from paddle3d.models.layers import FrozenBatchNorm2d, param_init
from paddle3d.apis import manager

__all__ = ["FPN", "LastLevelP6P7", "LastLevelP6"]


@manager.NECKS.add_component
class FPN(nn.Layer):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.

    This code is based on https://github.com/facebookresearch/detectron2/blob/333efcb6d0b60d7cceb7afc91bd96315cf211b0a/detectron2/modeling/backbone/fpn.py#L17
    """

    def __init__(self,
                 in_strides,
                 in_channels,
                 out_channel,
                 norm="",
                 top_block=None,
                 fuse_type="sum"):
        """
        Args:
            in_strides(list): strides list
            out_channel (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Layer or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN, self).__init__()
        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channel in enumerate(in_channels):
            if norm == "BN":
                lateral_norm = nn.BatchNorm2D(out_channel)
                output_norm = nn.BatchNorm2D(out_channel)
            elif norm == "FrozenBN":
                lateral_norm = FrozenBatchNorm2d(out_channel)
                output_norm = FrozenBatchNorm2d(out_channel)
            else:
                raise NotImplementedError()

            lateral_conv = [
                nn.Conv2D(
                    in_channel, out_channel, kernel_size=1, bias_attr=use_bias),
                lateral_norm
            ]
            output_conv = [
                nn.Conv2D(
                    out_channel,
                    out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=use_bias), output_norm
            ]

            stage = int(math.log2(in_strides[idx]))
            self.add_sublayer("fpn_lateral{}".format(stage),
                              nn.Sequential(*lateral_conv))
            self.add_sublayer("fpn_output{}".format(stage),
                              nn.Sequential(*output_conv))

            lateral_convs.append(nn.Sequential(*lateral_conv))
            output_convs.append(nn.Sequential(*output_conv))
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s
            for s in in_strides
        }
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2**(s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {
            k: out_channel
            for k in self._out_features
        }
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    def _init_weights(self):
        predictors = [self.lateral_convs, self.output_convs]
        for layers in predictors:
            for l in layers.sublayers():
                if isinstance(l, nn.Conv2D):
                    param_init.kaiming_uniform_init(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        param_init.constant_init(l.bias, value=0.0)

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        results = []
        prev_features = self.lateral_convs[0](x[-1])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
                zip(self.lateral_convs, self.output_convs)):
            if idx > 0:
                features = x[-idx - 1]
                top_down_features = F.interpolate(
                    prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = results[self._out_features.index(
                self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[
            i - 1], "Strides {} {} are not log2 contiguous".format(
                stride, strides[i - 1])


@manager.NECKS.add_component
class LastLevelP6P7(nn.Layer):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2D(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2D(out_channels, out_channels, 3, 2, 1)
        self._init_weights()

    def _init_weights(self):
        predictors = [self.p6, self.p7]
        for layers in predictors:
            param_init.kaiming_uniform_init(layers.weight, a=1)
            if layers.bias is not None:  # depth head may not have bias.
                param_init.constant_init(layers.bias, value=0.0)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@manager.NECKS.add_component
class LastLevelP6(nn.Layer):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_feature
        self.p6 = nn.Conv2D(in_channels, out_channels, 3, 2, 1)
        self._init_weights()

    def _init_weights(self):
        predictors = [self.p6]
        for layers in predictors:
            param_init.kaiming_uniform_init(layers.weight, a=1)
            if layers.bias is not None:  # depth head may not have bias.
                param_init.constant_init(layers.bias, value=0.0)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]
