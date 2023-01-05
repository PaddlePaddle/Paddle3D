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
from paddle3d.models.layers.param_init import (constant_init,
                                               xavier_uniform_init)
from paddle3d.models.voxel_encoders.pillar_encoder import build_norm_layer
from .transformer_layers import (FFN, BaseTransformerLayer, MultiHeadAttention,
                                 TransformerLayerSequence)


@manager.MODELS.add_component
class PETRTransformer(nn.Layer):
    """Implements the DETR transformer.
    Following the official DETR implementation.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    """

    def __init__(self,
                 decoder_embed_dims,
                 encoder=None,
                 decoder=None,
                 init_cfg=None,
                 cross=False):
        super(PETRTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = decoder_embed_dims  # self.decoder.embed_dims
        self.init_cfg = init_cfg
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.sublayers():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                reverse = False
                if isinstance(m, nn.Linear):
                    reverse = True

                xavier_uniform_init(m.weight, reverse=reverse)

                if hasattr(m, 'bias') and m.bias is not None:
                    constant_init(m.bias, value=0)

    def forward(self, x, mask, query_embed, pos_embed, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, c, h, w = x.shape

        # [bs, n, c, h, w] -> [bs, n*h*w, c]
        memory = x.transpose([0, 1, 3, 4, 2]).reshape([bs, -1, c])

        # [bs, n, c, h, w] -> [bs, n*h*w, c]
        pos_embed = pos_embed.transpose([0, 1, 3, 4, 2]).reshape([bs, -1, c])
        # [num_query, dim] -> [bs, num_query, dim]
        query_embed = query_embed.unsqueeze(0).tile([bs, 1, 1])
        # [bs, n, h, w] -> [bs, n*h*w]
        mask = mask.reshape([bs, 1, -1])
        target = paddle.zeros_like(query_embed)

        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            reg_branch=reg_branch,
        )
        memory = memory.reshape([n, h, w, bs, c]).transpose([3, 0, 4, 1, 2])
        return out_dec, memory


@manager.MODELS.add_component
class PETRDNTransformer(nn.Layer):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    """

    def __init__(self, embed_dims, encoder=None, decoder=None, cross=False):
        super(PETRDNTransformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.sublayers():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                reverse = False
                if isinstance(m, nn.Linear):
                    reverse = True

                xavier_uniform_init(m.weight, reverse=reverse)

    def forward(self,
                x,
                mask,
                query_embed,
                pos_embed,
                attn_masks=None,
                reg_branch=None):
        """Forward function for `Transformer`.
        """
        bs, n, c, h, w = x.shape
        memory = x.transpose([0, 1, 3, 4, 2]).reshape(
            [bs, -1, c])  # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.transpose([0, 1, 3, 4, 2]).reshape(
            [bs, -1, c])  # [bs, n, c, h, w] -> [n*h*w, bs, c]
        mask = mask.reshape([bs, 1, -1])  # [bs, n, h, w] -> [bs, n*h*w]
        target = paddle.zeros_like(query_embed)

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
        )
        memory = memory.reshape([n, h, w, bs, c]).transpose([3, 0, 4, 1, 2])
        return out_dec, memory


@manager.MODELS.add_component
class PETRTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    """

    def __init__(self,
                 attns,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LayerNorm'),
                 ffn_num_fcs=2,
                 use_recompute=True,
                 **kwargs):
        super(PETRTransformerDecoderLayer, self).__init__(
            attns=attns,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.use_recompute = use_recompute

    def _forward(
            self,
            query,
            key=None,
            value=None,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
    ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerDecoderLayer, self).forward(
            query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
        )

        return x

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
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_recompute and self.training:
            x = recompute(
                self._forward,
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
            )
        else:
            x = self._forward(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask)
        return x


@manager.MODELS.add_component
class PETRMultiheadAttention(nn.Layer):
    """A wrapper for ``paddle.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_prob=0.,
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(PETRMultiheadAttention, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.batch_first = batch_first

        self.attn = nn.MultiHeadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout = nn.Dropout(
            drop_prob) if drop_prob > 0. else nn.Identity()

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
            if attn_mask is None:
                attn_mask = ~key_padding_mask
                attn_mask = attn_mask.unsqueeze(1)  #.unsqueeze(1)
                out = self.attn(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                )
            else:
                raise NotImplementedError('key_padding_mask is not None')

        return identity + self.dropout(self.proj_drop(out))


class PETRTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.
    Args:
        post_norm (nn.Layer): normalization layer. Default:
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm=None, **kwargs):
        super(PETRTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm is not None:
            self.post_norm = post_norm
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@manager.MODELS.add_component
class PETRTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(PETRTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            # TODO hard code
            self.post_norm = nn.LayerNorm(self.embed_dims)
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)

        return paddle.stack(intermediate)
