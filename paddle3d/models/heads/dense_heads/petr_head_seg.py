# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.heads.dense_heads.target_assigner.hungarian_assigner import (
    HungarianAssigner3D, nan_to_num, normalize_bbox)
from paddle3d.models.layers import param_init
from paddle3d.models.layers.layer_libs import NormedLinear, inverse_sigmoid
from paddle3d.models.losses.focal_loss import FocalLoss, WeightedFocalLoss
from paddle3d.models.losses.weight_loss import WeightedL1Loss
from .samplers.pseudo_sampler import PseudoSampler
from .petr_head import reduce_mean, multi_apply, pos2posemb3d, SELayer, RegLayer


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = paddle.arange(num_pos_feats, dtype='int32')
    dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = paddle.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                         axis=-1).flatten(-2)
    pos_y = paddle.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                         axis=-1).flatten(-2)
    posemb = paddle.concat((pos_y, pos_x), axis=-1)
    return posemb


@manager.HEADS.add_component
class PETRHeadseg(nn.Layer):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    """

    def __init__(
            self,
            num_classes,
            in_channels,
            num_query=100,
            num_lane=100,
            num_reg_fcs=2,
            transformer=None,
            transformer_lane=None,
            sync_cls_avg_factor=False,
            positional_encoding=None,
            code_weights=None,
            bbox_coder=None,
            loss_cls=None,
            loss_bbox=None,
            loss_iou=None,
            loss_lane_mask=None,
            assigner=None,
            with_position=True,
            with_multiview=False,
            depth_step=0.8,
            depth_num=64,
            LID=False,
            depth_start=1,
            position_level=0,
            position_range=[-65, -65, -8.0, 65, 65, 8.0],
            group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
            init_cfg=None,
            normedlinear=False,
            with_se=False,
            with_time=False,
            with_detach=False,
            with_multi=False,
            **kwargs):

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2
            ]
        self.code_weights = self.code_weights[:self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        self.assigner = HungarianAssigner3D(
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

        self.sampler = PseudoSampler()

        self.num_query = num_query
        self.num_lane = num_lane
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs

        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = position_level
        self.with_position = with_position
        self.with_multiview = with_multiview

        self.num_pred = 6
        self.normedlinear = normedlinear
        self.with_se = with_se
        self.with_time = with_time
        self.with_detach = with_detach
        self.with_multi = with_multi
        self.group_reg_dims = group_reg_dims
        super(PETRHeadseg, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.loss_cls = loss_cls

        self.loss_bbox = loss_bbox
        self.loss_lane_mask = loss_lane_mask

        self.cls_out_channels = num_classes

        self.positional_encoding = positional_encoding

        initializer = paddle.nn.initializer.Assign(self.code_weights)
        self.code_weights = self.create_parameter(
            [len(self.code_weights)], default_initializer=initializer)
        self.code_weights.stop_gradient = True

        self.bbox_coder = bbox_coder
        self.pc_range = self.bbox_coder.point_cloud_range
        self._init_layers()
        self.transformer = transformer
        self.transformer_lane = transformer_lane
        self.pd_eps = paddle.to_tensor(np.finfo('float32').eps)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = nn.Conv2D(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = nn.Conv2D(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU())
        if self.normedlinear:
            cls_branch.append(
                NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        lane_branch = []
        for _ in range(self.num_reg_fcs):
            lane_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            lane_branch.append(nn.ReLU())
        lane_branch.append(nn.Linear(self.embed_dims, 768))
        lane_branch = nn.Sequential(*lane_branch)

        if self.with_multi:
            reg_branch = RegLayer(self.embed_dims, self.num_reg_fcs,
                                  self.group_reg_dims)
        else:
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
            reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.LayerList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_pred)])
        self.reg_branches = nn.LayerList(
            [copy.deepcopy(reg_branch) for _ in range(self.num_pred)])
        self.lane_branches = nn.LayerList(
            [copy.deepcopy(lane_branch) for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2D(
                    self.embed_dims * 3 // 2,
                    self.embed_dims * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2D(
                    self.embed_dims * 4,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2D(
                    self.embed_dims,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2D(
                    self.embed_dims,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2D(
                    self.position_dim,
                    self.embed_dims * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2D(
                    self.embed_dims * 4,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0),
            )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        if self.with_se:
            self.se = SELayer(self.embed_dims)
        nx = ny = round(math.sqrt(self.num_lane))
        x = (paddle.arange(nx) + 0.5) / nx
        y = (paddle.arange(ny) + 0.5) / ny
        xy = paddle.meshgrid(x, y)
        self.reference_points_lane = paddle.concat(
            [xy[0].reshape([-1])[..., None], xy[1].reshape([-1])[..., None]],
            -1)

        self.query_embedding_lane = nn.Sequential(
            nn.Linear(self.embed_dims * 2 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.input_proj.apply(param_init.reset_parameters)
        self.cls_branches.apply(param_init.reset_parameters)
        self.reg_branches.apply(param_init.reset_parameters)
        self.lane_branches.apply(param_init.reset_parameters)
        self.adapt_pos3d.apply(param_init.reset_parameters)

        if self.with_position:
            self.position_encoder.apply(param_init.reset_parameters)

        if self.with_se:
            self.se.apply(param_init.reset_parameters)

        self.transformer.init_weights()
        self.transformer_lane.init_weights()
        param_init.uniform_init(self.reference_points.weight, 0, 1)
        if self.loss_cls.use_sigmoid:
            bias_val = param_init.init_bias_by_prob(0.01)
            for m in self.cls_branches:
                param_init.constant_init(m[-1].bias, value=bias_val)

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        if getattr(self, 'in_export_mode', False):
            pad_h, pad_w = img_metas['image_shape']
        else:
            pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = paddle.arange(H, dtype='float32') * pad_h / H
        coords_w = paddle.arange(W, dtype='float32') * pad_w / W

        if self.LID:
            index = paddle.arange(
                start=0, end=self.depth_num, step=1, dtype='float32')
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = paddle.arange(
                start=0, end=self.depth_num, step=1, dtype='float32')
            bin_size = (
                self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        # W, H, D, 3
        coords = paddle.stack(paddle.meshgrid(
            [coords_w, coords_h, coords_d])).transpose([1, 2, 3, 0])
        coords = paddle.concat((coords, paddle.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * paddle.maximum(
            coords[..., 2:3],
            paddle.ones_like(coords[..., 2:3]) * eps)

        if not getattr(self, 'in_export_mode', False):
            img2lidars = []
            for img_meta in img_metas:
                img2lidar = []
                for i in range(len(img_meta['lidar2img'])):
                    img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
                img2lidars.append(np.asarray(img2lidar))

            img2lidars = np.asarray(img2lidars)

            # (B, N, 4, 4)
            img2lidars = paddle.to_tensor(img2lidars).astype(coords.dtype)
        else:
            img2lidars = img_metas['img2lidars']

        coords = coords.reshape([1, 1, W, H, D, 4]).tile(
            [B, N, 1, 1, 1, 1]).reshape([B, N, W, H, D, 4, 1])

        img2lidars = img2lidars.reshape([B, N, 1, 1, 1, 16]).tile(
            [1, 1, W, H, D, 1]).reshape([B, N, W, H, D, 4, 4])

        coords3d = paddle.matmul(img2lidars, coords)
        coords3d = coords3d.reshape(coords3d.shape[:-1])[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.astype('float32').flatten(-2).sum(-1) > (
            D * 0.5)
        coords_mask = masks | coords_mask.transpose([0, 1, 3, 2])

        coords3d = coords3d.transpose([0, 1, 4, 5, 3, 2]).reshape(
            [B * N, self.depth_num * 3, H, W])

        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.reshape([B, N, self.embed_dims, H,
                                                 W]), coords_mask

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats[self.position_level]

        batch_size, num_cams = x.shape[0], x.shape[1]

        if self.with_detach:
            current_frame = x[:, :6]
            past_frame = x[:, 6:]
            x = paddle.concat([current_frame, past_frame.detach()], 1)

        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = paddle.ones((batch_size, num_cams, input_img_h, input_img_w))

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = self.input_proj(x.flatten(0, 1))
        x = x.reshape([batch_size, num_cams, *x.shape[-3:]])

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).cast('bool')

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(
                mlvl_feats, img_metas, masks)

            if self.with_se:
                coords_position_embeding = self.se(
                    coords_position_embeding.flatten(0, 1),
                    x.flatten(0, 1)).reshape(x.shape)

            pos_embed = coords_position_embeding

            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).reshape(
                    x.shape)
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = paddle.concat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).reshape(
                    x.shape)
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).reshape(
                    x.shape)
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = paddle.concat(pos_embeds, 1)

        reference_points = self.reference_points.weight

        query_det = self.query_embedding(pos2posemb3d(reference_points))
        query_lane = self.query_embedding_lane(
            pos2posemb2d(self.reference_points_lane))

        reference_points = reference_points.unsqueeze(0).tile(
            [batch_size, 1, 1])

        outs_dec, _ = self.transformer(x, masks, query_det, pos_embed,
                                       self.reg_branches)
        outs_dec = nan_to_num(outs_dec)

        outs_dec_lane, _ = self.transformer_lane(x, masks, query_lane,
                                                 pos_embed, self.reg_branches)
        outs_dec_lane = nan_to_num(outs_dec_lane)
        lane_queries = outs_dec_lane

        if self.with_time:
            time_stamps = []
            for img_meta in img_metas:
                time_stamps.append(np.asarray(img_meta['timestamp']))

            time_stamp = paddle.to_tensor(time_stamps, dtype=x.dtype)
            time_stamp = time_stamp.reshape([batch_size, -1, 6])

            mean_time_stamp = (
                time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)

        outputs_classes = []
        outputs_coords = []
        outputs_lanes = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            outputs_lane = self.lane_branches[lvl](lane_queries[lvl])

            tmp[..., 0:2] += reference[..., 0:2]

            tmp[..., 0:2] = F.sigmoid(tmp[..., 0:2])
            tmp[..., 4:5] += reference[..., 2:3]

            tmp[..., 4:5] = F.sigmoid(tmp[..., 4:5])

            if self.with_time:
                tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_lanes.append(outputs_lane)

        all_cls_scores = paddle.stack(outputs_classes)
        all_bbox_preds = paddle.stack(outputs_coords)
        all_lane_preds = paddle.stack(outputs_lanes)

        all_bbox_preds[..., 0:1] = (
            all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) +
            self.pc_range[0])
        all_bbox_preds[..., 1:2] = (
            all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) +
            self.pc_range[1])
        all_bbox_preds[..., 4:5] = (
            all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) +
            self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'all_lane_preds': all_lane_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def export_forward(self, mlvl_feats, img_metas, time_stamp=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats[self.position_level]

        batch_size, num_cams = x.shape[0], x.shape[1]

        if self.with_detach:
            current_frame = x[:, :6]
            past_frame = x[:, 6:]
            x = paddle.concat([current_frame, past_frame.detach()], 1)

        input_img_h, input_img_w = img_metas['image_shape']

        masks = paddle.zeros([batch_size, num_cams, input_img_h, input_img_w])

        x = self.input_proj(x.flatten(0, 1))
        x = x.reshape([batch_size, num_cams, *x.shape[-3:]])

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).cast('bool')

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(
                mlvl_feats, img_metas, masks)

            if self.with_se:
                coords_position_embeding = self.se(
                    coords_position_embeding.flatten(0, 1),
                    x.flatten(0, 1)).reshape(x.shape)

            pos_embed = coords_position_embeding

            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).reshape(
                    x.shape)
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = paddle.concat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).reshape(
                    x.shape)
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).reshape(
                    x.shape)
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = paddle.concat(pos_embeds, 1)

        reference_points = self.reference_points.weight
        query_det = self.query_embedding(pos2posemb3d(reference_points))
        query_lane = self.query_embedding_lane(
            pos2posemb2d(self.reference_points_lane))

        reference_points = reference_points.unsqueeze(0).tile(
            [batch_size, 1, 1])

        outs_dec, _ = self.transformer(x, masks, query_det, pos_embed,
                                       self.reg_branches)
        outs_dec = nan_to_num(outs_dec)

        outs_dec_lane, _ = self.transformer_lane(x, masks, query_lane,
                                                 pos_embed, self.reg_branches)
        outs_dec_lane = nan_to_num(outs_dec_lane)
        lane_queries = outs_dec_lane

        if self.with_time:
            time_stamp = time_stamp.reshape([batch_size, -1, 6])
            mean_time_stamp = (
                time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)

        outputs_classes = []
        outputs_coords = []
        outputs_lanes = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            outputs_lane = self.lane_branches[lvl](lane_queries[lvl])

            tmp[..., 0:2] += reference[..., 0:2]

            tmp[..., 0:2] = F.sigmoid(tmp[..., 0:2])
            tmp[..., 4:5] += reference[..., 2:3]

            tmp[..., 4:5] = F.sigmoid(tmp[..., 4:5])

            if self.with_time:
                tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_lanes.append(outputs_lane)

        all_cls_scores = paddle.stack(outputs_classes)
        all_bbox_preds = paddle.stack(outputs_coords)
        all_lane_preds = paddle.stack(outputs_lanes)

        all_bbox_preds[..., 0:1] = (
            all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) +
            self.pc_range[0])
        all_bbox_preds[..., 1:2] = (
            all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) +
            self.pc_range[1])
        all_bbox_preds[..., 4:5] = (
            all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) +
            self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'all_lane_preds': all_lane_preds,
        }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.shape[0]
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = paddle.full((num_bboxes, ), self.num_classes, dtype='int64')

        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = paddle.ones([num_bboxes])

        # bbox targets
        code_size = gt_bboxes.shape[1]
        bbox_targets = paddle.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = paddle.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        if sampling_result.pos_gt_bboxes.shape[1] == 4:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes.reshape(
                sampling_result.pos_gt_bboxes.shape[0], self.code_size - 1)
        else:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    lane_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_lane_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.shape[0]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = paddle.concat(labels_list, 0)
        label_weights = paddle.concat(label_weights_list, 0)
        bbox_targets = paddle.concat(bbox_targets_list, 0)
        bbox_weights = paddle.concat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape([-1, self.cls_out_channels])
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                paddle.to_tensor([cls_avg_factor], dtype=cls_scores.dtype))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels,
                                 label_weights) / (cls_avg_factor + self.pd_eps)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = paddle.to_tensor([num_total_pos], dtype=loss_cls.dtype)
        num_total_pos = paddle.clip(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape([-1, bbox_preds.shape[-1]])
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        # paddle.all
        isnotnan = paddle.isfinite(normalized_bbox_targets).all(axis=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan], normalized_bbox_targets[isnotnan],
            bbox_weights[isnotnan]) / (num_total_pos + self.pd_eps)

        lane_preds = lane_preds.squeeze(0)

        loss_lane_mask = self.loss_lane_mask(lane_preds, gt_lane_list[0])
        loss_lane_mask = nan_to_num(loss_lane_mask)

        loss_cls = nan_to_num(loss_cls)
        loss_bbox = nan_to_num(loss_bbox)

        return loss_cls, loss_bbox, loss_lane_mask

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_lanes,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        all_lane_preds = preds_dicts['all_lane_preds']

        num_dec_layers = len(all_cls_scores)

        def get_gravity_center(bboxes):
            bottom_center = bboxes[:, :3]
            gravity_center = np.zeros_like(bottom_center)
            gravity_center[:, :2] = bottom_center[:, :2]
            gravity_center[:, 2] = bottom_center[:, 2] + bboxes[:, 5] * 0.5
            return gravity_center

        gt_bboxes_list = [
            paddle.concat((paddle.to_tensor(get_gravity_center(gt_bboxes)),
                           paddle.to_tensor(gt_bboxes[:, 3:])),
                          axis=1) for gt_bboxes in gt_bboxes_list
        ]
        gt_lanes = [gt_lanes[0]]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_lanes_list = [gt_lanes for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, losses_lane_masks = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_lane_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_lanes_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                paddle.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_mask'] = losses_lane_masks[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_mask_i in zip(
                losses_cls[:-1], losses_bbox[:-1], losses_lane_masks[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            num_dec_layer += 1
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
