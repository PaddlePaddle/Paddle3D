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
# Modified from BEVFormer (https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.layers.param_init import (constant_init,
                                               xavier_uniform_init)
from paddle3d.models.transformers.utils import masked_fill
from paddle3d.ops import ms_deform_attn
from paddle3d.utils.logger import logger


@manager.ATTENTIONS.add_component
class SpatialCrossAttention(nn.Layer):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 batch_first=False,
                 deformable_attention=dict(
                     type_name='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs):
        super(SpatialCrossAttention, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        deformable_attention_ = copy.deepcopy(deformable_attention)
        layer_name = deformable_attention_.pop('type_name')
        attention_layer = manager.ATTENTIONS.components_dict[layer_name]
        self.deformable_attention = attention_layer(**deformable_attention_)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    @paddle.no_grad()
    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_uniform_init(self.output_proj.weight, reverse=True)
        constant_init(self.output_proj.bias, value=0)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        query = query.cast(paddle.float32)
        key = key.cast(paddle.float32)
        value = value.cast(paddle.float32)
        if query_pos is not None:
            query_pos = query_pos.cast(paddle.float32)
        if reference_points_cam is not None:
            reference_points_cam = reference_points_cam.cast(paddle.float32)

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = paddle.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        D = reference_points_cam.shape[3]
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = paddle.max(bev_mask.any(-1).sum(-1))
        max_len = max_len.numpy()
        #max_len = 2500
        #max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = paddle.zeros(
            [bs, self.num_cams, max_len, self.embed_dims], dtype=query.dtype)
        reference_points_rebatch = paddle.zeros(
            [bs, self.num_cams, max_len, D, 2],
            dtype=reference_points_cam.dtype)

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                #queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                queries_rebatch[j, i, :len(index_query_per_img
                                           )] = paddle.gather(
                                               query[j], index_query_per_img)
                #reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(
                    index_query_per_img)] = paddle.gather(
                        reference_points_per_img[j], index_query_per_img)

        num_cams, l, bs, embed_dims = key.shape

        key = key.transpose([2, 0, 1, 3]).reshape(
            [bs * self.num_cams, l, self.embed_dims])
        value = value.transpose([2, 0, 1, 3]).reshape(
            [bs * self.num_cams, l, self.embed_dims])

        queries = self.deformable_attention(
            query=queries_rebatch.reshape(
                [bs * self.num_cams, max_len, self.embed_dims]),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.reshape(
                [bs * self.num_cams, max_len, D, 2]),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index).reshape(
                [bs, self.num_cams, max_len, self.embed_dims])
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                #slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
                bs_idx = paddle.full_like(
                    index_query_per_img, j, dtype=index_query_per_img.dtype)
                scatter_index = paddle.stack(
                    [bs_idx, index_query_per_img]).transpose([1, 0])
                slots = paddle.scatter_nd_add(
                    slots, scatter_index,
                    queries[j, i, :len(index_query_per_img)])

        count = bev_mask.sum(-1) > 0
        count = count.transpose([1, 2, 0]).sum(-1)
        count = paddle.clip(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@manager.ATTENTIONS.add_component
class MSDeformableAttention3D(nn.Layer):
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
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True):
        super(MSDeformableAttention3D, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

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
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

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
            [self.num_heads, 1, 1,
             2]).tile([1, self.num_levels, self.num_points, 1])

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.set_value(grid_init.reshape([-1]))
        constant_init(self.attention_weights.weight, value=0)
        constant_init(self.attention_weights.bias, value=0)
        xavier_uniform_init(self.value_proj.weight, reverse=True)
        constant_init(self.value_proj.bias, value=0)

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
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
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
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.transpose([1, 0, 2])
            value = value.transpose([1, 0, 2])

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = masked_fill(value, key_padding_mask[..., None], 0.0)
        value = value.reshape([bs, num_value, self.num_heads, -1])
        sampling_offsets = self.sampling_offsets(query).reshape([
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        ])
        attention_weights = self.attention_weights(query).reshape(
            [bs, num_query, self.num_heads, self.num_levels * self.num_points])

        attention_weights = F.softmax(attention_weights, -1)

        attention_weights = attention_weights.reshape(
            [bs, num_query, self.num_heads, self.num_levels, self.num_points])

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = paddle.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points.reshape(
                [bs, num_query, 1, 1, 1, num_Z_anchors, xy])
            sampling_offsets = sampling_offsets / \
                offset_normalizer.reshape([1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1]])
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.reshape([
                bs, num_query, num_heads, num_levels,
                num_all_points // num_Z_anchors, num_Z_anchors, xy
            ])
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.reshape(
                [bs, num_query, num_heads, num_levels, num_all_points, xy])

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        # sampling_locations.stop_gradient = True
        # attention_weights.stop_gradient = True
        output = ms_deform_attn.ms_deform_attn(
            value, sampling_locations, attention_weights, spatial_shapes,
            level_start_index, self.im2col_step)

        if not self.batch_first:
            output = output.transpose([1, 0, 2])

        return output


@manager.ATTENTIONS.add_component
class CustomMSDeformableAttention(nn.Layer):
    """An attention module used in Deformable-Detr.

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
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None):
        super(CustomMSDeformableAttention, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            logger.warnings(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
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
            [self.num_heads, 1, 1,
             2]).tile([1, self.num_levels, self.num_points, 1])

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
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query, embed_dims)
            query = query.transpose([1, 0, 2])
            value = value.transpose([1, 0, 2])

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape([bs, num_value, self.num_heads, -1])

        sampling_offsets = self.sampling_offsets(query).reshape([
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        ])
        attention_weights = self.attention_weights(query).reshape(
            [bs, num_query, self.num_heads, self.num_levels * self.num_points])
        attention_weights = F.softmax(attention_weights, -1)

        attention_weights = attention_weights.reshape(
            [bs, num_query, self.num_heads, self.num_levels, self.num_points])
        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            sampling_locations = reference_points.reshape([bs, num_query, 1, num_Z_anchors, 1, xy]) \
                + sampling_offsets \
                / offset_normalizer.reshape([1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1]])
        elif reference_points.shape[-1] == 4:
            unsqueeze_reference_points = paddle.unsqueeze(
                reference_points, axis=[2, 4])[..., :2]
            sampling_locations = unsqueeze_reference_points \
                + sampling_offsets / self.num_points \
                * unsqueeze_reference_points \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        value = value.cast(paddle.float32)
        sampling_locations = sampling_locations.cast(paddle.float32)
        output = ms_deform_attn.ms_deform_attn(
            value, sampling_locations, attention_weights, spatial_shapes,
            level_start_index, self.im2col_step)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.transpose([1, 0, 2])

        return self.dropout(output) + identity
