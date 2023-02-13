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
# This code is based on https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py#L407
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy

import paddle
import paddle.nn as nn

from paddle3d.apis import manager
from paddle3d.models.layers.param_init import (constant_init,
                                               xavier_uniform_init)
from paddle3d.utils.logger import logger


@manager.ATTENTIONS.add_component
class MultiheadAttention(nn.Layer):
    """A wrapper for ``paddle.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type_name='Dropout', p=0.),
                 batch_first=False,
                 **kwargs):
        super(MultiheadAttention, self).__init__()
        if 'dropout' in kwargs:
            logger.warning('The arguments `dropout` in MultiheadAttention '
                           'has been deprecated, now you can separately '
                           'set `attn_drop`(float), proj_drop(float), '
                           'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['p'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiHeadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        if dropout_layer:
            dropout_layer_ = copy.deepcopy(dropout_layer)
            dropout_layer_name = dropout_layer_.pop("type_name")
            self.dropout_layer = getattr(nn,
                                         dropout_layer_name)(**dropout_layer_)
        else:
            self.dropout_layer = nn.Identity()
        self.init_weights()

    @paddle.no_grad()
    def init_weights(self):
        for layer in self.attn.sublayers():
            if isinstance(layer, nn.Linear):
                xavier_uniform_init(layer.weight, reverse=True)
                constant_init(layer.bias, value=0)

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
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
                [num_queries, bs, embed_dims]
                if self.batch_first is False, else
                [bs, num_queries embed_dims].
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
                    logger.warning(f'position encoding of key is'
                                   f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``paddle.nn.MultiheadAttention`` is (batch, num_query, embed_dims)
        # We should adjust the shape of dataflow from
        # num_query_first (num_query ,batch, embed_dims) to batch_first
        # (batch, num_query, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if not self.batch_first:
            query = query.transpose([1, 0, 2])
            key = key.transpose([1, 0, 2])
            value = value.transpose([1, 0, 2])

        if key_padding_mask is None:
            if attn_mask is not None:
                attn_mask = ~attn_mask
            out = self.attn(
                query=query, key=key, value=value, attn_mask=attn_mask)
        else:
            if attn_mask is None:
                attn_mask = ~key_padding_mask
                out = self.attn(
                    query=query, key=key, value=value, attn_mask=attn_mask)
            else:
                raise ValueError('key_padding_mask is not None')

        if not self.batch_first:
            out = out.transpose([1, 0, 2])

        return identity + self.dropout_layer(self.proj_drop(out))
