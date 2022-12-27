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

# ------------------------------------------------------------------------
# Modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/positional_encoding.py
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Uniform

from paddle3d.apis import manager


@manager.POSITIONAL_ENCODING.add_component
class LearnedPositionalEncoding(nn.Layer):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(
            row_num_embed,
            num_feats,
            weight_attr=ParamAttr(initializer=Uniform(0, 1)))
        self.col_embed = nn.Embedding(
            col_num_embed,
            num_feats,
            weight_attr=ParamAttr(initializer=Uniform(0, 1)))
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = paddle.arange(w)
        y = paddle.arange(h)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = paddle.concat((x_embed.unsqueeze(0).tile([h, 1, 1]),
                             y_embed.unsqueeze(1).tile([1, w, 1])),
                            axis=-1).transpose([2, 0, 1]).unsqueeze(0).tile(
                                [mask.shape[0], 1, 1, 1])
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str
