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
# Modified from BEVFormer (https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.layers.param_init import (constant_init,
                                               xavier_uniform_init)
from paddle3d.models.transformers.utils import masked_fill
from paddle3d.ops import ms_deform_attn
from paddle3d.utils import logger


@manager.ATTENTIONS.add_component
class TemporalSelfAttention(nn.Layer):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None):

        super(TemporalSelfAttention, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            logger.warning(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    @paddle.no_grad()
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets.weight, value=0.)
        constant_init(self.sampling_offsets.bias, value=0.)
        thetas = paddle.arange(
            self.num_heads,
            dtype=paddle.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)).reshape(
            [self.num_heads, 1, 1, 2]).tile(
                [1, self.num_levels * self.num_bev_queue, self.num_points, 1])

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.set_value(grid_init.reshape([-1]))
        constant_init(self.attention_weights.weight, value=0)
        constant_init(self.attention_weights.bias, value=0)
        xavier_uniform_init(self.value_proj.weight, reverse=True)
        constant_init(self.value_proj.bias, value=0)
        xavier_uniform_init(self.output_proj.weight, reverse=True)
        constant_init(self.output_proj.bias, value=0)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = paddle.stack([query, query],
                                 1).reshape([bs * 2, len_bev, c])

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query, embed_dims)
            query = query.transpose([1, 0, 2])
            value = value.transpose([1, 0, 2])
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        query = paddle.concat([value[:bs], query], -1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = masked_fill(value, key_padding_mask[..., None], 0.0)

        value = value.reshape(
            [bs * self.num_bev_queue, num_value, self.num_heads, -1])

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape([
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels,
            self.num_points, 2
        ])
        attention_weights = self.attention_weights(query).reshape([
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels * self.num_points
        ])
        attention_weights = F.softmax(attention_weights, -1)

        attention_weights = attention_weights.reshape([
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels,
            self.num_points
        ])

        attention_weights = attention_weights.transpose([0, 3, 1, 2, 4, 5])\
            .reshape([bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points])
        sampling_offsets = sampling_offsets.transpose([0, 3, 1, 2, 4, 5, 6])\
            .reshape([bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2])

        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        # sampling_locations.stop_gradient = True
        # attention_weights.stop_gradient = True
        value = value.cast(paddle.float32)
        sampling_locations = sampling_locations.cast(paddle.float32)
        output = ms_deform_attn.ms_deform_attn(
            value, sampling_locations, attention_weights, spatial_shapes,
            level_start_index, self.im2col_step)

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.transpose([1, 2, 0])

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.reshape([num_query, embed_dims, bs, self.num_bev_queue])
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.transpose([2, 0, 1])

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.transpose([1, 0, 2])

        return self.dropout(output) + identity
