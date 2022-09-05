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


class ScoreNet(nn.Layer):
    def __init__(self, in_channel, out_channel, hidden_unit=[16],
                 last_bn=False):
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.LayerList()
        self.mlp_bns_hidden = nn.LayerList()

        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs_nohidden = nn.Conv2D(
                in_channel, out_channel, 1, bias_attr=not last_bn)
            if self.last_bn:
                self.mlp_bns_nohidden = nn.BatchNorm2D(out_channel)

        else:
            self.mlp_convs_hidden.append(
                nn.Conv2D(in_channel, hidden_unit[0], 1,
                          bias_attr=False))  # from in_channel to first hidden
            self.mlp_bns_hidden.append(nn.BatchNorm2D(hidden_unit[0]))
            for i in range(1, len(hidden_unit)
                           ):  # from 2nd hidden to next hidden to last hidden
                self.mlp_convs_hidden.append(
                    nn.Conv2D(
                        hidden_unit[i - 1], hidden_unit[i], 1, bias_attr=False))
                self.mlp_bns_hidden.append(nn.BatchNorm2D(hidden_unit[i]))
            self.mlp_convs_hidden.append(
                nn.Conv2D(
                    hidden_unit[-1], out_channel, 1,
                    bias_attr=not last_bn))  # from last hidden to out_channel
            self.mlp_bns_hidden.append(nn.BatchNorm2D(out_channel))

    def forward(self, xyz, calc_scores='softmax', bias_attr=0):
        B, _, N, K = xyz.shape
        scores = xyz

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))
            else:
                scores = self.mlp_convs_nohidden(scores)
        else:
            for i, conv in enumerate(self.mlp_convs_hidden):
                if i == len(self.mlp_convs_hidden
                            ) - 1:  # if the output layer, no ReLU
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))
                    else:
                        scores = conv(scores)
                else:
                    bn = self.mlp_bns_hidden[i]
                    scores = F.relu(bn(conv(scores)))

        if calc_scores == 'softmax':
            scores = F.softmax(
                scores, axis=1
            ) + bias_attr  # B*m*N*K, where bias may bring larger gradient
        elif calc_scores == 'sigmoid':
            scores = F.sigmoid(scores) + bias_attr  # B*m*N*K
        else:
            raise ValueError('Not Implemented!')

        scores = paddle.transpose(scores, [0, 2, 3, 1])

        return scores
