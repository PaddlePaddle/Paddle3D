# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import warnings
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle3d.apis import manager


@manager.LOSSES.add_component
class SigmoidCeLoss(nn.Layer):
    def __init__(self, loss_weight=1.0):
        super(SigmoidCeLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        """Forward function to calculate accuracy.

        Args:
            pred (paddle.Tensor): Prediction of models.
            target (paddle.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        pos_weight = (targets == 0).cast('float32').sum(
            axis=1) / (targets == 1).cast('float32').sum(axis=1).clip(min=1.0)
        pos_weight = pos_weight.unsqueeze(1)
        weight_loss = targets * pos_weight + (1 - targets)
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="mean", weight=weight_loss)
        return self.loss_weight * loss
