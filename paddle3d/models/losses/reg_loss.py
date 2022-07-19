# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.models.layers.layer_libs import _transpose_and_gather_feat


class RegLoss(nn.Layer):
    """
    This function refers to https://github.com/tianweiy/CenterPoint/blob/cb25e870b271fe8259e91c5d17dcd429d74abc91/det3d/models/losses/centernet_loss.py#L6
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.cast('float32').unsqueeze(2)

        loss = F.l1_loss(pred * mask, target * mask, reduction='none')
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose([2, 1, 0]).sum(axis=2).sum(axis=1)
        return loss
