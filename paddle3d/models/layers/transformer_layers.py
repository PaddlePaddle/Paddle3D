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

# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy
import math
import warnings
from typing import Sequence

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute

from paddle3d.apis import manager
from paddle3d.models.layers.layer_libs import act_layer_from_config
from paddle3d.models.layers.param_init import (constant_init,
                                               xavier_uniform_init)


class FFN(nn.Layer):
    """Implements feed-forward networks (FFNs) with identity connection.
    """

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU'),
                 ffn_drop=0.,
                 dropout_prob=0.,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super(FFN, self).__init__()

        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.act_layer = act_layer_from_config(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.act_layer, nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        self.dropout_layer = nn.Dropout(
            dropout_prob) if dropout_prob else nn.Identity()

        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(x)

        if identity is None:
            identity = x

        return identity + self.dropout_layer(out)


class BaseTransformerLayer(nn.Layer):
    """Base `TransformerLayer` for vision transformer.
    """

    def __init__(self,
                 attns=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU'),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LayerNorm'),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):

        super(BaseTransformerLayer, self).__init__()

        self.batch_first = batch_first
        if 'feedforward_channels' in kwargs:
            ffn_cfgs['feedforward_channels'] = kwargs['feedforward_channels']

        if 'ffn_dropout' in kwargs:
            ffn_cfgs['ffn_drop'] = kwargs['ffn_dropout']

        assert set(operation_order) & set(
            ['self_attn', 'norm', 'ffn', 'cross_attn']) == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')

        assert num_attn == len(attns), f'The length ' \
            f'of attn_cfg {num_attn} is ' \
            f'not consistent with the number of attention' \
            f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.LayerList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                attns[index].batch_first = self.batch_first
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attns[index].operation_name = operation_name
                self.attentions.append(attns[index])
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = nn.LayerList()
        num_ffns = operation_order.count('ffn')

        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(FFN(**ffn_cfgs[ffn_index]))

        self.norms = nn.LayerList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            # TODO hard code
            self.norms.append(nn.LayerNorm(self.embed_dims))

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, paddle.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)

                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


class TransformerLayerSequence(nn.Layer):
    """Base class for TransformerEncoder and TransformerDecoder in vision
    transformer.

    As base-class of Encoder and Decoder in vision transformer.
    Support customization such as specifying different kind
    of `transformer_layer` in `transformer_coder`.

    Args:
        transformerlayer: paddle.nn.Layer. Default: None.
        num_layers (int): The number of `TransformerLayer`. Default: None.

    """

    def __init__(self, transformerlayers=None, num_layers=None):
        super(TransformerLayerSequence, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.LayerList()
        self.layers.append(transformerlayers)
        for i in range(num_layers - 1):
            self.layers.append(copy.deepcopy(transformerlayers))

        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.
        """
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        return query


@manager.MODELS.add_component
class MultiHeadAttention(nn.Layer):
    """A wrapper for ``paddle.nn.MultiheadAttention``.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_prob=0.,
                 batch_first=True,
                 **kwargs):
        super(MultiHeadAttention, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        # only support batch first
        self.batch_first = True

        self.attn = nn.MultiHeadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)

        self.dropout_layer = nn.Dropout(
            drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiHeadAttention`.
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if key_padding_mask is None:
            if attn_mask is not None:
                attn_mask = ~attn_mask
            out = self.attn(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
            )
        else:
            raise NotImplementedError(
                'key_padding_mask is not None not support now')

        return identity + self.dropout_layer(self.proj_drop(out))
