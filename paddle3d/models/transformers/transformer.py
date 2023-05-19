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
# Modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.layers.param_init import (
    constant_init, normal_init, reset_parameters, xavier_uniform_init)
from paddle3d.models.transformers.utils import rotate
from .attentions import CustomMSDeformableAttention, MSDeformableAttention3D, TemporalSelfAttention

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


@manager.TRANSFORMERS.add_component
class PerceptionTransformer(nn.Layer):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.init_layers()
        self.rotate_center = rotate_center
        self.init_weights()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        level_embeds = self.create_parameter((self.num_feature_levels,
                                              self.embed_dims))
        self.add_parameter('level_embeds', level_embeds)
        cams_embeds = self.create_parameter((self.num_cams, self.embed_dims))
        self.add_parameter('cams_embeds', cams_embeds)
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_sublayer('norm', nn.LayerNorm(self.embed_dims))

    @paddle.no_grad()
    def init_weights(self):
        """Initialize the transformer weights."""
        normal_init(self.level_embeds)
        normal_init(self.cams_embeds)
        xavier_uniform_init(self.reference_points.weight, reverse=True)
        constant_init(self.reference_points.bias, value=0)
        for layer in self.can_bus_mlp:
            if isinstance(layer, nn.Linear):
                reset_parameters(layer)
                # xavier_uniform_init(layer.weight, reverse=True)
            elif isinstance(layer, nn.LayerNorm):
                constant_init(layer.weight, value=1)
                constant_init(layer.bias, value=0)

    def get_bev_features(self,
                         mlvl_feats,
                         bev_queries,
                         bev_h,
                         bev_w,
                         grid_length=[0.512, 0.512],
                         bev_pos=None,
                         prev_bev=None,
                         **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].shape[0]
        bev_queries = bev_queries.unsqueeze(1).tile([1, bs, 1])
        bev_pos = bev_pos.flatten(2).transpose([2, 0, 1])
        '''
        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = paddle.to_tensor(
            [shift_x, shift_y]).transpose([1, 0])  # xy, bs -> bs, xy

        '''
        delta_x = paddle.concat(
            [each['can_bus'][0] for each in kwargs['img_metas']])
        delta_y = paddle.concat(
            [each['can_bus'][1] for each in kwargs['img_metas']])
        #np.save('delta_x.npy', delta_x.numpy())
        #np.save('delta_y.npy', delta_y.numpy())
        ego_angle = paddle.concat(
            [each['can_bus'][-2] / np.pi for each in kwargs['img_metas']])
        #np.save('ego_angle.npy', ego_angle)
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = paddle.sqrt(delta_x**2 + delta_y**2)
        #np.save('translation_length.npy', translation_length.numpy())
        translation_angle = paddle.atan2(delta_y, delta_x) / np.pi
        # translation_angle = paddle.angle(delta_y, delta_x) / pi_tensor
        #np.save('translation_angle.npy', translation_angle.numpy())
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            paddle.cos(bev_angle * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            paddle.sin(bev_angle * np.pi) / grid_length_x / bev_w
        #np.save('shift_x.npy', shift_x.numpy())
        #np.save('shift_y.npy', shift_y.numpy())

        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift

        shift = paddle.stack([shift_x,
                              shift_y]).transpose([1, 0])  # xy, bs -> bs, xy

        shift = shift.cast(bev_queries.dtype)
        '''
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.transpose([1, 0, 2])
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape([
                        bev_h, bev_w, -1]).transpose([2, 0, 1])
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    # #np.save('tmp_prev_bev.npy', tmp_prev_bev.numpy())
                    tmp_prev_bev = tmp_prev_bev.transpose([1, 2, 0]).reshape([
                        bev_h * bev_w, 1, -1])
                    prev_bev[:, i] = tmp_prev_bev[:, 0]
        '''
        if prev_bev is not None:
            if self.rotate_prev_bev:
                if not getattr(self, 'export_model', False):
                    for i in range(bs):
                        valid_prev_bev = prev_bev[:, i].cast('bool').any().cast(
                            'int32')
                        rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                        tmp_prev_bev = prev_bev[:, i].reshape(
                            [bev_h, bev_w, -1]).transpose([2, 0, 1])
                        tmp_prev_bev = rotate(
                            tmp_prev_bev,
                            rotation_angle,
                            center=self.rotate_center)
                        tmp_prev_bev = tmp_prev_bev.transpose(
                            [1, 2, 0]).reshape([bev_h * bev_w, 1, -1])
                        prev_bev_new = tmp_prev_bev[:,
                                                    0] * valid_prev_bev + prev_bev[:, i] * (
                                                        1 - valid_prev_bev)
                        prev_bev = prev_bev_new.unsqueeze(1)
                else:
                    valid_prev_bev = prev_bev[:, 0].cast('bool').any().cast(
                        'int32')
                    rotation_angle = kwargs['img_metas'][0]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, 0].reshape(
                        [bev_h, bev_w, -1]).transpose([2, 0, 1])
                    tmp_prev_bev = rotate(
                        tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.transpose([1, 2, 0]).reshape(
                        [bev_h * bev_w, 1, -1])
                    prev_bev_new = tmp_prev_bev[:,
                                                0] * valid_prev_bev + prev_bev[:, 0] * (
                                                    1 - valid_prev_bev)
                    prev_bev = prev_bev_new.unsqueeze(1)

        # add can bus signals
        # can_bus = paddle.to_tensor(
        #     [each['can_bus'] for each in kwargs['img_metas']], dtype=bev_queries.dtype)  # [:, :]
        can_bus = paddle.stack([
            each['can_bus'] for each in kwargs['img_metas']
        ]).cast(bev_queries.dtype)
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).transpose([1, 0, 3, 2])
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].cast(
                    feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].cast(
                feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = paddle.concat(feat_flatten, 2)
        tmp_spatial_shapes = paddle.to_tensor(
            spatial_shapes, dtype=paddle.int64, place=paddle.CPUPlace())
        spatial_shapes = paddle.to_tensor(tmp_spatial_shapes)
        level_start_index = paddle.concat((paddle.zeros(
            (1, ), dtype=paddle.int64), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.transpose(
            [0, 2, 1, 3])  # (num_cam, H*W, bs, embed_dims)

        # #np.save('bev_queries.npy', bev_queries.numpy())
        # #np.save('feat_flatten.npy', feat_flatten.numpy())
        # #np.save('bev_pos.npy', bev_pos.numpy())
        # #np.save('prev_bev.npy', prev_bev.numpy())
        # #np.save('spatial_shapes.npy', spatial_shapes.numpy())
        # #np.save('level_start_index.npy', level_start_index.numpy())
        # #np.save('shift.npy', shift.numpy())
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs)
        # #np.save('bev_embed.npy', bev_embed.numpy())
        # exit()
        return bev_embed

    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].shape[0]
        query_pos, query = paddle.split(
            object_query_embed, [self.embed_dims, self.embed_dims], axis=1)
        query_pos = query_pos.unsqueeze(0).expand([bs, -1, -1])
        query = query.unsqueeze(0).expand([bs, -1, -1])
        reference_points = self.reference_points(query_pos)
        reference_points = F.sigmoid(reference_points)
        init_reference_out = reference_points

        query = query.transpose([1, 0, 2])
        query_pos = query_pos.transpose([1, 0, 2])
        bev_embed = bev_embed.transpose([1, 0, 2])

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=paddle.concat([
                paddle.full([1, 1], bev_h, dtype=paddle.int64),
                paddle.full([1, 1], bev_w, dtype=paddle.int64)
            ],
                                         axis=-1),
            level_start_index=paddle.full([1], 0, dtype=paddle.int64),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out


@manager.TRANSFORMERS.add_component
class RTEBevTransformer(nn.Layer):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_center=[100, 100],
                 **kwargs):
        super(RTEBevTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.init_layers()
        self.rotate_center = rotate_center
        self.init_weights()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        pass

    @paddle.no_grad()
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if len(p.shape) > 1:
                if len(p.shape) == 2:
                    xavier_uniform_init(p, reverse=True)
                else:
                    xavier_uniform_init(p)
                # nn.init.xavier_uniform_(p)
        for m in self.sublayers():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

        pass

    def forward(self,
                bev_feat,
                query_pos, 
                query,
                reference_points,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bs = bev_feat.shape[0]

        bev_embed = bev_feat.flatten(2)
        bev_embed = bev_embed.transpose([2, 0, 1])

        init_reference_out = reference_points

        query_pos = query_pos.transpose([1, 0, 2])
        if query is None:
            query = paddle.zeros_like(query_pos)
        else:
            query = query.transpose([1, 0, 2])

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=paddle.to_tensor([[bev_h, bev_w]],
                                            dtype=paddle.int64),
            level_start_index=paddle.to_tensor([0], dtype=paddle.int64),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out