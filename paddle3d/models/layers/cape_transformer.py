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
from typing import List
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute

from paddle3d.apis import manager
from paddle3d.models.layers import param_init
from paddle3d.models.layers.param_init import (constant_init,
                                               xavier_uniform_init)
from paddle3d.models.voxel_encoders.pillar_encoder import build_norm_layer
from .transformer_layers import (FFN, BaseTransformerLayer, MultiHeadAttention,
                                 TransformerLayerSequence)
from paddle3d.models.layers.layer_libs import NormedLinear, inverse_sigmoid


class QcR_Modulation(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.scale_emb = nn.Sequential(
            nn.Linear(9, dim), nn.LayerNorm(dim), nn.Sigmoid())

    def forward(self, x, R):
        bs, num_cam = R.shape[:2]
        R = R.flatten(2)
        scale_emb = self.scale_emb(R)
        x = x[:, None].tile([1, num_cam, 1, 1])
        x = x * scale_emb[:, :, None]
        return x


class V_R_Modulation(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.scale_emb = nn.Sequential(
            nn.Linear(9, dim), nn.LayerNorm(dim), nn.Sigmoid())

    def forward(self, feature, R):
        bs, num_cam = R.shape[:2]
        R = R.flatten(2)
        scale_emb = self.scale_emb(R)
        feature = feature * scale_emb[:, :, None]
        return feature


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = paddle.arange(num_pos_feats, dtype='int32')
    dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = paddle.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                         axis=-1).flatten(-2)
    pos_y = paddle.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                         axis=-1).flatten(-2)
    pos_z = paddle.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()),
                         axis=-1).flatten(-2)
    posemb = paddle.concat((pos_y, pos_x, pos_z), axis=-1)
    return posemb


class SELayer(nn.Layer):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class Ego_emb(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.ego_emb = nn.Sequential(
            nn.Linear(9, dim), nn.LayerNorm(dim), nn.Sigmoid())

    def forward(self, img_metas, x):
        ego_matrix = self.get_curlidar2prevlidar(
            img_metas, x)[:3, :3][None, None]  # 1, 1, 3, 3
        ego_emb = self.ego_emb(ego_matrix.flatten(2))  # 1, 1, d
        return ego_emb

    def get_curlidar2prevlidar(self, img_metas, x):
        '''
            get ego motion matrix in lidar axis.
            cur_lidar----->prev cam------>prev_lidar.
            curlidar2prevcam @ prevcam2prevlidar =  curlidar2prevcam @ curcam2curlidar = curlidar2prevcam @ inverse(curlidar2curcam)

        '''
        curlidar2prevcam = paddle.to_tensor(
            img_metas[0]['extrinsics'][6].T, dtype='float32')  # (4, 4)
        curlidar2curcam = paddle.to_tensor(
            img_metas[0]['extrinsics'][0].T, dtype='float32')  # (4, 4)
        prevcam2prevlidar = paddle.inverse(curlidar2curcam)
        return (prevcam2prevlidar @ curlidar2prevcam)


class MLP_Fusion(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj_k_a = nn.Linear(dim, dim)
        self.proj_k_b = nn.Linear(dim, dim)
        self.proj_v_a = nn.Linear(dim, dim)
        self.proj_v_b = nn.Linear(dim, dim)
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.Sigmoid())
        self.ego_emb = Ego_emb(dim)

    def forward(self, a, b, img_metas):
        '''
            a: (b Q d)
            b: (b Q d)
        '''
        k_a = self.proj_k_a(a)
        k_b = self.proj_k_b(b)
        ego_emb = self.ego_emb(img_metas, k_b)
        ego_k_b = k_b * ego_emb
        w = self.fc(paddle.concat([k_a, ego_k_b], -1))
        v_a = self.proj_v_a(a)
        v_b = self.proj_v_b(b)
        a = w * v_a
        b = (1 - w) * v_b
        return a, b


@manager.MODELS.add_component
class CrossAttention(nn.Layer):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head

        init_val = [dim_head**(-0.5) for _ in range(self.heads)]
        initializer = paddle.nn.initializer.Assign(init_val)

        self.proj_dict = nn.LayerDict({
            'q_g':
            nn.Linear(dim, heads * dim_head, bias_attr=qkv_bias),
            'k_g':
            nn.Linear(dim, heads * dim_head, bias_attr=qkv_bias),
            'q_a':
            nn.Linear(dim, heads * dim_head, bias_attr=qkv_bias),
            'k_a':
            nn.Linear(dim, heads * dim_head, bias_attr=qkv_bias),
            'v':
            nn.Linear(dim, heads * dim_head, bias_attr=qkv_bias)
        })

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

        self.scale_a = self.create_parameter([len(init_val)],
                                             default_initializer=initializer)
        self.scale_g = self.create_parameter([len(init_val)],
                                             default_initializer=initializer)

    def forward(self, k_g, q_g, k_a, q_a, v, mask):
        """
        k_g: (b n K d)
        q_g: (b n Q d)
        k_a: (b n K d)
        q_a: (b Q d)
        v:   (b n K d)
        mask: (b n K)
        """

        b, n, Q, d = q_g.shape

        skip = q_a
        # Project with multiple heads
        k_g = self.proj_dict['k_g'](k_g)  # b n K (heads dim_head)
        q_g = self.proj_dict['q_g'](q_g)  # b n Q (heads dim_head)
        k_a = self.proj_dict['k_a'](k_a)  # b n K (heads dim_head)
        q_a = self.proj_dict['q_a'](q_a)  # b Q (heads dim_head)
        v = self.proj_dict['v'](v)  # b n K (heads dim_head)
        q_a = q_a[:, None].expand([b, n, Q, d])

        # Group the head dim with batch dim
        bb, nn, hw, c = k_g.shape
        k_g = k_g.reshape([bb, nn, hw, self.heads, self.dim_head]).transpose(
            [0, 3, 1, 2, 4]).reshape([-1, nn, hw, self.dim_head])
        q_g = q_g.reshape([b, n, Q, self.heads, self.dim_head]).transpose(
            [0, 3, 1, 2, 4]).reshape([-1, nn, Q, self.dim_head])
        k_a = k_a.reshape([bb, nn, hw, self.heads, self.dim_head]).transpose(
            [0, 3, 1, 2, 4]).reshape([-1, nn, hw, self.dim_head])
        q_a = q_a.reshape([b, n, Q, self.heads, self.dim_head]).transpose(
            [0, 3, 1, 2, 4]).reshape([-1, nn, Q, self.dim_head])
        v = v.reshape([bb, nn, hw, self.heads, self.dim_head]).transpose(
            [0, 3, 1, 2, 4]).reshape([-1, nn * hw, self.dim_head])

        # Dot product attention along cameras
        dot_g = paddle.einsum('b n Q d, b n K d -> b n Q K', q_g, k_g)
        dot_a = paddle.einsum('b n Q d, b n K d -> b n Q K', q_a, k_a)
        dot_g = dot_g.reshape([b, self.heads, n, Q, hw])
        dot_a = dot_a.reshape([b, self.heads, n, Q, hw])

        dot_a = dot_a * self.scale_a.unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1).unsqueeze(0)
        dot_g = dot_g * self.scale_g.unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1).unsqueeze(0)
        dot = dot_a + dot_g  # b, m, n, Q, K

        def masked_fill(x, mask, value):
            y = paddle.full(x.shape, value, x.dtype)
            return paddle.where(mask, y, x)

        dot = masked_fill(dot, mask[:, None, :, None, :], float('-inf'))
        dot = dot.transpose([0, 1, 3, 2, 4]).reshape([b * self.heads, Q, -1])
        att = F.softmax(dot, axis=-1)

        # Combine values (image level features).
        a = paddle.einsum('b Q K, b K d -> b Q d', att, v)
        a = a.reshape([b, self.heads, Q, self.dim_head]).transpose(
            [0, 2, 1, 3]).reshape([b, Q, self.heads * self.dim_head])
        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + skip

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)

        return z


class MLP(nn.Layer):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@manager.MODELS.add_component
class CAPETransformer(nn.Layer):
    """Implements the DETR transformer.
    Following the official DETR implementation.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    """

    def __init__(self,
                 num_cameras: int,
                 num_layers: int,
                 feat_dim: int,
                 feat_stride: int,
                 image_height: int,
                 image_width: int,
                 bound: List[float],
                 with_fpe: bool,
                 depth_start: int,
                 depth_num: int,
                 att_layer=None,
                 tf_layer=dict(groups=30, heads=8, hidden_dim=512),
                 scalar=10,
                 noise_scale=1.0,
                 noise_trans=0.0,
                 num_classes=10,
                 with_time=True):
        super(CAPETransformer, self).__init__()
        self.num_cameras = num_cameras
        self.num_queries = att_layer.num_queries
        self.hidden_dim = att_layer.hidden_dim
        self.feat_dim = feat_dim
        self.with_time = with_time

        ys, xs = paddle.meshgrid(
            paddle.arange(feat_stride / 2, image_height, feat_stride),
            paddle.arange(feat_stride / 2, image_width, feat_stride))
        image_plane = paddle.stack(
            [xs, ys, paddle.ones_like(xs)], axis=-1).flatten(0, 1).astype(
                'float32')  # hw * 3
        self.register_buffer('image_plane', image_plane, persistable=True)

        self.register_buffer(
            'bound', paddle.to_tensor(bound).reshape([2, 3]), persistable=True)

        self.reference_points = nn.Embedding(self.num_queries, 3)
        param_init.uniform_init(self.reference_points.weight, 0, 1)

        if self.with_time:
            self.mf = nn.LayerList(
                [MLP_Fusion(tf_layer['hidden_dim']) for _ in range(num_layers)])

        self.cva_layers = nn.LayerList(
            [copy.deepcopy(att_layer) for _ in range(num_layers)])
        self.cva_layers[0].conditional = None

        self.content_prior = nn.Embedding(self.num_queries, self.hidden_dim)

        #TODO: use camera PE
        self.camera_embedding = nn.Embedding(num_cameras, self.hidden_dim)

        self.bev_embed = MLP(self.hidden_dim * 3 // 2, self.hidden_dim,
                             self.hidden_dim, 2)

        self.feature_linear = nn.Linear(feat_dim, self.hidden_dim)

        self.with_fpe = with_fpe
        if self.with_fpe:
            self.fpe = SELayer(self.hidden_dim)

        self.depth_start = depth_start
        self.depth_num = depth_num
        # point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.cam_position_range = [
            bound[1], -bound[5], self.depth_start, bound[4], -bound[2], bound[3]
        ]
        self.register_buffer(
            'cam_bound',
            paddle.to_tensor(self.cam_position_range).reshape([2, 3]),
            persistable=True)

        self.position_dim = 3 * self.depth_num
        self.position_encoder = nn.Sequential(
            nn.Conv2D(
                self.position_dim,
                self.hidden_dim * 4,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(),
            nn.Conv2D(
                self.hidden_dim * 4,
                self.hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0),
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 // 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.dn_query_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim * 3 // 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.QcR = QcR_Modulation(self.hidden_dim)
        self.V_R = V_R_Modulation(self.hidden_dim)

        self.num_classes = num_classes
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.split = 0.75
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

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

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:

            def get_gravity_center(bboxes):
                bottom_center = bboxes[:, :3]
                gravity_center = np.zeros_like(bottom_center)
                gravity_center[:, :2] = bottom_center[:, :2]
                gravity_center[:, 2] = bottom_center[:, 2] + bboxes[:, 5] * 0.5
                return gravity_center

            targets = [
                paddle.concat(
                    (paddle.to_tensor(
                        get_gravity_center(img_meta['gt_bboxes_3d'])),
                     paddle.to_tensor(img_meta['gt_bboxes_3d'][:, 3:])),
                    axis=1) for img_meta in img_metas
            ]
            labels = [img_meta['gt_labels_3d'] for img_meta in img_metas]
            known = [(paddle.ones_like(t)) for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = paddle.concat(known)
            known_num = [t.shape[0] for t in targets]
            labels = paddle.concat([t for t in labels])
            boxes = paddle.concat([t for t in targets])
            batch_idx = paddle.concat(
                [paddle.full((t.shape[0], ), i) for i, t in enumerate(targets)])

            known_indice = paddle.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.reshape([-1])
            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.tile([self.scalar, 1]).reshape([-1])
            known_labels = labels.tile([self.scalar,
                                        1]).reshape([-1]).astype('int64')
            known_bid = batch_idx.tile([self.scalar, 1]).reshape([-1])
            known_bboxs = boxes.tile([self.scalar, 1])
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = paddle.rand(known_bbox_center.shape) * 2 - 1.0
                known_bbox_center += paddle.multiply(
                    rand_prob, diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (
                    known_bbox_center[..., 0:1] - self.pc_range[0]) / (
                        self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (
                    known_bbox_center[..., 1:2] - self.pc_range[1]) / (
                        self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (
                    known_bbox_center[..., 2:3] - self.pc_range[2]) / (
                        self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clip(min=0.0, max=1.0)
                mask = paddle.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = paddle.zeros([pad_size, 3])
            padded_reference_points = paddle.concat(
                [padding_bbox, reference_points], axis=0).unsqueeze(0).tile(
                    [batch_size, 1, 1])

            if len(known_num):
                map_known_indice = paddle.concat(
                    [paddle.to_tensor(list(range(num))) for num in known_num])
                map_known_indice = paddle.concat([
                    map_known_indice + single_pad * i
                    for i in range(self.scalar)
                ]).astype('int64')
            if len(known_bid):
                padded_reference_points[(known_bid.astype('int64'),
                                         map_known_indice)] = known_bbox_center

            tgt_size = pad_size + self.num_query
            attn_mask = paddle.ones([tgt_size, tgt_size]) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad *
                              (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad *
                              i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad *
                              (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad *
                              i] = True

            mask_dict = {
                'known_indice':
                paddle.to_tensor(known_indice, dtype='int64'),
                'batch_idx':
                paddle.to_tensor(batch_idx, dtype='int64'),
                'map_known_indice':
                paddle.to_tensor(map_known_indice, dtype='int64'),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx':
                know_idx,
                'pad_size':
                pad_size
            }

        else:
            padded_reference_points = reference_points.unsqueeze(0).tile(
                [batch_size, 1, 1])
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def prepare_emb(self, feature, mask, I_inv, R_inv, t, ref_points, img_metas,
                    seq_length):
        img_embed, _ = self.position_embeding(feature, I_inv, img_metas, mask)
        img_embed = img_embed.flatten(-2).transpose([0, 1, 3, 2])
        bs, nc, d, h, w = feature.shape  #[:2]
        ref_points_unormalized = ref_points * (
            self.bound[1] - self.bound[0]) + self.bound[0]

        feature = feature.reshape([bs * nc, d, h * w]).transpose([0, 2, 1])
        feature = self.feature_linear(feature)
        feature = feature.reshape([bs, nc, h * w, -1])

        if self.with_fpe:
            img_embed = self.fpe(img_embed, feature)

        mask = mask.reshape([bs, nc, -1])
        # ref_points_unormalized.shape: bs N 3
        ref_points_unormalized = ref_points_unormalized[:, None].tile(
            [seq_length, 1, 1, 1])  # 2 * bs 1 Q 3
        R = paddle.inverse(R_inv)

        world = ref_points_unormalized + t[:, :, None]  # b n Q 3
        world = (R[:, :, None] @ world[..., None]).squeeze(-1)
        bev_embed = self.bev_embed(pos2posemb3d(
            world, self.hidden_dim // 2))  # b n Q d
        return feature, mask, img_embed, bev_embed, R

    def position_embeding(self, img_feats, I_inv, img_metas, masks=None):
        eps = 1e-5

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        B, N, C, H, W = img_feats.shape
        coords_h = paddle.arange(H, dtype='float32') * pad_h / H
        coords_w = paddle.arange(W, dtype='float32') * pad_w / W

        index = paddle.arange(
            start=0, end=self.depth_num, step=1, dtype='float32')
        index_1 = index + 1
        bin_size = (self.cam_position_range[5] - self.depth_start) / (
            self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1

        D = coords_d.shape[0]
        # W, H, D, 3
        coords = paddle.stack(paddle.meshgrid(
            [coords_w, coords_h, coords_d])).transpose([1, 2, 3, 0])
        coords[..., :2] = coords[..., :2] * paddle.maximum(
            coords[..., 2:3],
            paddle.ones_like(coords[..., 2:3]) * eps)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))

        coords = coords.reshape([1, 1, W, H, D, 3]).tile(
            [B, N, 1, 1, 1, 1]).reshape([B, N, W, H, D, 3, 1])
        I_inv = I_inv.reshape([B, N, 1, 1, 1, 9]).tile(
            [1, 1, W, H, D, 1]).reshape([B, N, W, H, D, 3, 3])

        coords3d = paddle.matmul(I_inv, coords)
        coords3d = coords3d.reshape(coords3d.shape[:-1])[..., :3]
        coords3d[..., 0:1] = (
            coords3d[..., 0:1] - self.cam_position_range[0]) / (
                self.cam_position_range[3] - self.cam_position_range[0])
        coords3d[..., 1:2] = (
            coords3d[..., 1:2] - self.cam_position_range[1]) / (
                self.cam_position_range[4] - self.cam_position_range[1])
        coords3d[..., 2:3] = (
            coords3d[..., 2:3] - self.cam_position_range[2]) / (
                self.cam_position_range[5] - self.cam_position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.astype('float32').flatten(-2).sum(-1) > (
            D * 0.5)
        coords_mask = masks | coords_mask.transpose([0, 1, 3, 2])

        coords3d = coords3d.transpose([0, 1, 4, 5, 3, 2]).reshape(
            [B * N, self.depth_num * 3, H, W])

        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.reshape([B, N, self.hidden_dim, H,
                                                 W]), coords_mask

    def forward(self,
                feature,
                mask,
                I_inv,
                R_inv,
                t,
                img_metas,
                return_prev_query=False,
                mean_time_stamp=0.5):
        return_list = []
        return_prev_list = []
        bs, nc, d, h, w = feature.shape  #[:2]
        f = 2 if self.with_time else 1

        feature = feature.reshape([-1, self.num_cameras, d, h, w])
        mask = mask.reshape([-1, self.num_cameras, h, w])

        I_inv = I_inv.reshape([-1, self.num_cameras, *I_inv.shape[-2:]])
        R_inv = R_inv.reshape([-1, self.num_cameras, *R_inv.shape[-2:]])

        t = t.reshape([-1, self.num_cameras, t.shape[-2] * t.shape[-1]])
        x = self.content_prior.weight.unsqueeze(0).tile([bs, 1, 1])

        ref_points = self.reference_points.weight  # N, 3
        ref_points, attn_mask, mask_dict = self.prepare_for_dn(
            bs, ref_points, img_metas)  # bs N 3
        if self.training:
            pad_size = mask_dict['pad_size']
            dn_query = self.dn_query_embedding(
                pos2posemb3d(ref_points[:, :pad_size, :],
                             self.hidden_dim // 2))  #.repeat(bs, 1, 1)
            x = paddle.concat([dn_query, x], 1)  # bs N 256

        lidar_obj_pe = self.query_embedding(
            pos2posemb3d(ref_points,
                         self.hidden_dim // 2))  #.repeat(bs, 1, 1) # bs, N, 256
        cam_pe = self.camera_embedding.weight.tile([bs, 1, 1])

        if self.with_time:
            cur_x = x
            prev_x = x
            lidar_obj_pe = paddle.concat([lidar_obj_pe, lidar_obj_pe],
                                         0)  # 2 * bs, N, 256
            cam_pe = paddle.concat([cam_pe, cam_pe], 0)

            feature, mask, img_embed, bev_embed, R = self.prepare_emb(
                feature, mask, I_inv, R_inv, t, ref_points, img_metas, f)
            for mf, cva in zip(self.mf, self.cva_layers):
                x = paddle.concat([cur_x, prev_x], 0)  # 2 * bs N 256
                modulated_x = self.QcR(x, R)  # b num_cam Q d
                modulated_v = self.V_R(feature, R)
                x = cva(x, modulated_x, lidar_obj_pe, modulated_v, cam_pe, mask,
                        img_embed, bev_embed, attn_mask)
                cur_x, prev_x = paddle.split(x, [bs, bs])  # bs N 256
                # cur_x, prev_x = x[0:1, ...], x[1:2, ...]
                cur_x, prev_x = mf(cur_x, prev_x, img_metas)
                return_list.append(cur_x)
                return_prev_list.append(prev_x)
            if not return_prev_query:
                return paddle.stack(return_list), ref_points
            else:
                return paddle.stack(return_list), paddle.stack(
                    return_prev_list), ref_points, mask_dict
        else:
            feature, mask, img_embed, bev_embed, R = self.prepare_emb(
                feature, mask, I_inv, R_inv, t, ref_points, img_metas, f)
            for cva in self.cva_layers:
                modulated_x = self.QcR(x, R)  # b num_cam Q d
                modulated_v = self.V_R(feature, R)
                x = cva(x, modulated_x, lidar_obj_pe, modulated_v, cam_pe, mask,
                        img_embed, bev_embed, attn_mask)
                return_list.append(x)
            return paddle.stack(return_list), ref_points, mask_dict


@manager.MODELS.add_component
class CrossViewAttention(nn.Layer):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    """

    def __init__(self,
                 num_queries: int,
                 hidden_dim: int,
                 qkv_bias: bool,
                 heads: int = 4,
                 dim_head: int = 32,
                 conditional: bool = True):
        super(CrossViewAttention, self).__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.cross_attend = CrossAttention(hidden_dim, heads, dim_head,
                                           qkv_bias)
        self.conditional = MLP(hidden_dim, hidden_dim, hidden_dim,
                               2) if conditional else None
        self.sl_layer = nn.MultiHeadAttention(hidden_dim, heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x, modulated_x, lidar_obj_pe, feature, camera_pe, mask,
                img_embed, bev_embed, attn_mask):
        """
        x: (b, Q, d)
        obj_pe: (b, Q, d)
        feature: (b, n, K, d)
        camera_pe: (b, n, d)
        mask: (b, n, K)
        img_embed: (b, n, K, d)
        bev_embed: (b, n, Q, d)

        Returns: (b, d, H, W)
        """
        b, n, _, _ = feature.shape

        if self.conditional is not None:
            bev_embed = self.conditional(modulated_x) * bev_embed

        val = feature
        k_a, q_a = feature + camera_pe[:, :, None], x + lidar_obj_pe
        if self.training:
            updated_x = recompute(self.cross_attend, img_embed, bev_embed, k_a,
                                  q_a, val, mask)
        else:
            updated_x = self.cross_attend(img_embed, bev_embed, k_a, q_a, val,
                                          mask)

        q = k = updated_x + lidar_obj_pe
        if attn_mask is not None:
            attn_mask = ~attn_mask
        tgt = self.sl_layer(q, k, value=updated_x, attn_mask=attn_mask)

        return self.norm1(updated_x + self.dropout1(tgt))
