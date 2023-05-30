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
# Modified from BEVFormer (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math
import copy
from functools import partial

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal

from paddle3d.apis import manager
from paddle3d.models.layers.param_init import constant_init, reset_parameters
from paddle3d.models.transformers.transformer import inverse_sigmoid
from paddle3d.models.transformers.utils import nan_to_num
from paddle3d.utils import dtype2float32
from paddle3d.utils.box import normalize_bbox
from paddle3d.models.layers import param_init
from paddle3d.models.heads.dense_heads.petr_head import multi_apply, reduce_mean, pos2posemb3d



@manager.HEADS.add_component
class RTEBevHead(nn.Layer):
    """Head of RTEBev.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 transformer,
                 positional_encoding=None,
                 num_query=100,
                 num_reg_fcs=2,
                 num_cls_fcs=2,
                 sync_cls_avg_factor=False,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_iou=None,
                 assigner=dict(
                     type='HungarianAssigner',
                     cls_cost=dict(type='ClassificationCost', weight=1.),
                     reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                     iou_cost=dict(type='IoUCost', iou_mode='giou',
                                   weight=2.0)),
                 sampler=None,
                 with_box_refine=False,
                 as_two_stage=False,
                 bbox_coder=None,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 use_clone=False,
                 num_queries_one2one=900,
                 k_one2many=0,
                 lambda_one2many=1.0,
                 **kwargs):
        super(RTEBevHead, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
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
        # self.code_weights = paddle.to_tensor(self.code_weights)
        initializer = paddle.nn.initializer.Assign(self.code_weights)
        self.code_weights = self.create_parameter(
            [len(self.code_weights)], default_initializer=initializer)
        self.code_weights.stop_gradient = True

        self.bbox_coder = bbox_coder
        self.point_cloud_range = self.bbox_coder.point_cloud_range
        self.real_w = self.point_cloud_range[3] - self.point_cloud_range[0]
        self.real_h = self.point_cloud_range[4] - self.point_cloud_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        assert loss_cls.loss_weight == assigner.cls_cost.weight, \
            'The classification weight for loss and matcher should be' \
            'exactly the same.'
        assert loss_bbox.loss_weight == assigner.reg_cost.weight, \
            'The regression L1 weight for loss and matcher ' \
            'should be exactly the same.'
        assert loss_iou.loss_weight == assigner.iou_cost.weight, \
            'The regression iou weight for loss and matcher should be' \
            'exactly the same.'
        self.assigner = assigner
        # DETR sampling=False, so use PseudoSampler
        self.sampler = sampler
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.fp16_enabled = False
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_iou = loss_iou
        self.use_clone = use_clone
        self.num_queries_one2one = num_queries_one2one
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = getattr(transformer, 'act_cfg', dict(type='ReLU'))
        self.activate = getattr(nn, self.act_cfg.pop('type'))()
        self.positional_encoding = positional_encoding
        self.transformer = transformer
        self.embed_dims = self.transformer.embed_dims

        if positional_encoding is not None:
            num_feats = positional_encoding.num_feats
            assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                f' and {num_feats}.'

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU())
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.LayerList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.LayerList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.LayerList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            if self.k_one2many > 0:
                self.reference_points = nn.Embedding(self.num_queries_one2one, 3)
                self.reference_points_12m = nn.Embedding(self.num_query - self.num_queries_one2one, 3)
            else:
                self.reference_points = nn.Embedding(self.num_query, 3)
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )

    @paddle.no_grad()
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for cls_layerlist in self.cls_branches:
            for cls_layer in cls_layerlist:
                if isinstance(cls_layer, nn.Linear):
                    reset_parameters(cls_layer)
                elif isinstance(cls_layer, nn.LayerNorm):
                    constant_init(cls_layer.weight, value=1)
                    constant_init(cls_layer.bias, value=0)

        for reg_layerlist in self.reg_branches:
            for reg_layer in reg_layerlist:
                if isinstance(reg_layer, nn.Linear):
                    reset_parameters(reg_layer)

        if self.loss_cls.use_sigmoid:
            prior_prob = 0.01
            bias_init = float(-np.log((1 - prior_prob) / prior_prob))
            for m in self.cls_branches:
                constant_init(m[-1].bias, value=bias_init)
        
        param_init.uniform_init(self.reference_points.weight, 0, 1)
        if self.k_one2many > 0:
            param_init.uniform_init(self.reference_points_12m.weight, 0, 1)

    def forward(self, mlvl_feats, img_metas=None, prev_bev=None, only_bev=False):
        if hasattr(self, 'amp_cfg_'):
            for key, mlvl_feat in enumerate(mlvl_feats):
                mlvl_feats[key] = mlvl_feat.cast(paddle.float16)

        bev_feat = mlvl_feats[0]
        bs, _, _, _ = bev_feat.shape
        
        dtype = mlvl_feats[0].dtype

        bev_pos = None
        bev_mask = None
        if self.positional_encoding is not None:
            bev_mask = paddle.zeros((bs, self.bev_h, self.bev_w), dtype=dtype)
            bev_pos = self.positional_encoding(bev_mask).cast(dtype)

        reference_points = self.reference_points.weight
        if self.k_one2many > 0 and self.training:
            reference_points_12m = self.reference_points_12m.weight
            reference_points = paddle.concat([reference_points, reference_points_12m], 0)

        reference_points = reference_points.unsqueeze(0).expand([bs, -1, -1])
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))

        # attn mask
        self_attn_mask = None
        if self.k_one2many > 0 and self.training:
            self_attn_mask = (
                paddle.zeros([self.num_query, self.num_query], dtype='bool')
            )
            self_attn_mask[self.num_queries_one2one :, 0 : self.num_queries_one2one] = True
            self_attn_mask[0 : self.num_queries_one2one, self.num_queries_one2one :] = True
            self_attn_mask = ~self_attn_mask

        outputs = self.transformer(
            bev_feat,
            query_embeds,
            None,
            reference_points.clip(0, 1),
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                            self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None, # noqa:E501
            img_metas=img_metas,
            attn_masks=self_attn_mask,
            prev_bev=None)

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.transpose([0, 2, 1, 3])
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            
            if self.use_clone:
                reference = inverse_sigmoid(reference.clone())
            else:
                reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = F.sigmoid(tmp[..., 0:2])
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = F.sigmoid(tmp[..., 4:5])
            tmp[..., 0:1] = (
                tmp[..., 0:1] *
                (self.point_cloud_range[3] - self.point_cloud_range[0]) +
                self.point_cloud_range[0])
            tmp[..., 1:2] = (
                tmp[..., 1:2] *
                (self.point_cloud_range[4] - self.point_cloud_range[1]) +
                self.point_cloud_range[1])
            tmp[..., 4:5] = (
                tmp[..., 4:5] *
                (self.point_cloud_range[5] - self.point_cloud_range[2]) +
                self.point_cloud_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = paddle.stack(outputs_classes)
        outputs_coords = paddle.stack(outputs_coords)

        all_cls_scores_one2many = None
        all_bbox_preds_one2many = None
        if self.k_one2many > 0 and self.training:
            all_cls_scores_one2one = outputs_classes[:, :, 0 : self.num_queries_one2one, :]
            all_bbox_preds_one2one = outputs_coords[:, :, 0 : self.num_queries_one2one, :]
            all_cls_scores_one2many = outputs_classes[:, :, self.num_queries_one2one :, :]
            all_bbox_preds_one2many = outputs_coords[:, :, self.num_queries_one2one :, :]

            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': all_cls_scores_one2one,
                'all_bbox_preds': all_bbox_preds_one2one,
                'all_cls_scores_one2many': all_cls_scores_one2many,
                'all_bbox_preds_one2many': all_bbox_preds_one2many,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'dn_mask_dict': None
            }
        else:
            outs = {
                'bev_embed': bev_feat,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
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
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = paddle.full((num_bboxes, ),
                             self.num_classes,
                             dtype=paddle.int64)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = paddle.ones((num_bboxes, ))

        # bbox targets
        bbox_targets = paddle.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = paddle.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
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

        labels_list, label_weights_list, bbox_targets_list, \
        bbox_weights_list, pos_inds_list, neg_inds_list = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
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
        cls_scores = cls_scores.reshape((-1, self.cls_out_channels))
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(paddle.to_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = paddle.to_tensor([num_total_pos], dtype=loss_cls.dtype)
        num_total_pos = paddle.clip(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape([-1, bbox_preds.shape[-1]])
        normalized_bbox_targets = normalize_bbox(bbox_targets,
                                                 self.point_cloud_range)
        isnotnan = paddle.isfinite(normalized_bbox_targets).all(axis=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan],
            normalized_bbox_targets[isnotnan],
            bbox_weights[isnotnan],
            avg_factor=num_total_pos)
        loss_cls = nan_to_num(loss_cls)
        loss_bbox = nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
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
        all_cls_scores = dtype2float32(all_cls_scores)
        all_bbox_preds = dtype2float32(all_bbox_preds)
        enc_cls_scores = dtype2float32(enc_cls_scores)
        enc_bbox_preds = dtype2float32(enc_bbox_preds)
        if self.k_one2many > 0:
            all_cls_scores_one2many = preds_dicts['all_cls_scores_one2many']
            all_bbox_preds_one2many = preds_dicts['all_bbox_preds_one2many']

        num_dec_layers = len(all_cls_scores)

        bboxes_list = []
        for gt_bboxes in gt_bboxes_list:
            bottom_center = gt_bboxes[:, :3]
            gravity_center = paddle.zeros_like(bottom_center)
            gravity_center[:, :2] = bottom_center[:, :2]
            gravity_center[:, 2] = bottom_center[:, 2] + gt_bboxes[:, 5] * 0.5
            bboxes_list.append(
                paddle.concat([gravity_center, gt_bboxes[:, 3:]], axis=-1))

        all_gt_bboxes_list = [bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_bboxes_ignore_list)

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

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        # for one2many
        if self.k_one2many > 0:
            one2many_gt_bboxes_list = []
            one2many_gt_labels_list = []
            for gt_bboxes in bboxes_list:
                one2many_gt_bboxes_list.append(gt_bboxes.tile([self.k_one2many, 1]))

            for gt_labels in gt_labels_list:
                one2many_gt_labels_list.append(gt_labels.tile([self.k_one2many]))

            all_gt_bboxes_list_one2many = [one2many_gt_bboxes_list for _ in range(num_dec_layers)]
            all_gt_labels_list_one2many = [one2many_gt_labels_list for _ in range(num_dec_layers)]
            all_gt_bboxes_ignore_list_one2many = all_gt_bboxes_ignore_list
            # one2many losses
            losses_cls_one2many, losses_bbox_one2many = multi_apply(
                self.loss_single, all_cls_scores_one2many, all_bbox_preds_one2many,
                all_gt_bboxes_list_one2many, all_gt_labels_list_one2many, 
                all_gt_bboxes_ignore_list_one2many)

            # loss for one for many
            loss_dict['loss_cls_12m'] = losses_cls_one2many[-1] * self.lambda_one2many
            loss_dict['loss_bbox_12m'] = losses_bbox_one2many[-1] * self.lambda_one2many

            num_dec_layer = 0
            for loss_cls_i_12m, loss_bbox_i_12m in zip(losses_cls_one2many[:-1],
                                            losses_bbox_one2many[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls_12m'] = loss_cls_i_12m
                loss_dict[f'd{num_dec_layer}.loss_bbox_12m'] = loss_bbox_i_12m
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
        for key, value in preds_dicts.items():
            preds_dicts[key] = dtype2float32(value)
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
