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

from paddle3d.apis import manager
from paddle3d.datasets.semantic_kitti.semantic_kitti import \
    SemanticKITTIDataset

__all__ = ["SSGLossComputation"]


@manager.LOSSES.add_component
class SSGLossComputation(nn.Layer):
    """
    Loss layer of SqueezeSegV3.

    Args:
        num_classes: Number of classes.
        epsilon_w: Epsilon for weight normalization.
        ignore_index: Index of ignored class.
    """

    def __init__(self,
                 num_classes: int,
                 epsilon_w: float,
                 ignore_index: int = None):
        super().__init__()

        remap_lut = SemanticKITTIDataset.build_remap_lut()
        content = paddle.zeros([num_classes], dtype="float32")
        for cl, freq in SemanticKITTIDataset.CONTENT.items():
            x_cl = remap_lut[cl]
            content[x_cl] += freq
        weight = 1. / (content + epsilon_w)
        if ignore_index in range(num_classes):
            weight[ignore_index] = 0.

        self.loss_func = nn.NLLLoss(weight, ignore_index=ignore_index)

    def forward(self, logits_list, target):
        loss_list = []
        for logits in logits_list:
            loss = self.loss_func(
                paddle.log(paddle.clip(logits, min=1e-8)),
                F.interpolate(target, logits.shape[-2:],
                              mode="nearest").squeeze(axis=1))
            loss_list.append(loss)

        return sum(loss_list)
