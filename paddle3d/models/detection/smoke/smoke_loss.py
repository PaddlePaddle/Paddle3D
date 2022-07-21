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
"""
This code is based on https://github.com/lzccccc/SMOKE/blob/master/smoke/modeling/heads/smoke_head/loss.py
Ths copyright is MIT License
"""

import copy
import os

import cv2
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn import functional as F

from paddle3d.apis import manager
from paddle3d.models.detection.smoke.smoke_coder import SMOKECoder
from paddle3d.models.layers import select_point_of_interest
from paddle3d.models.losses import FocalLoss


@manager.LOSSES.add_component
class SMOKELossComputation(object):
    """Convert targets and preds to heatmaps&regs, compute
       loss with CE and L1
    """

    def __init__(self,
                 depth_ref,
                 dim_ref,
                 reg_loss="DisL1",
                 loss_weight=(1., 10.),
                 max_objs=50):

        self.smoke_coder = SMOKECoder(depth_ref, dim_ref)
        self.cls_loss = FocalLoss(alpha=2, beta=4)
        self.reg_loss = reg_loss
        self.loss_weight = loss_weight
        self.max_objs = max_objs

    def prepare_targets(self, targets):
        """get heatmaps, regressions and 3D infos from targets
        """

        heatmaps = targets["hm"]
        regression = targets["reg"]
        cls_ids = targets["cls_ids"]
        proj_points = targets["proj_p"]
        dimensions = targets["dimensions"]
        locations = targets["locations"]
        rotys = targets["rotys"]
        trans_mat = targets["trans_mat"]
        K = targets["K"]
        reg_mask = targets["reg_mask"]
        flip_mask = targets["flip_mask"]
        bbox_size = targets["bbox_size"]
        c_offsets = targets["c_offsets"]

        return heatmaps, regression, dict(
            cls_ids=cls_ids,
            proj_points=proj_points,
            dimensions=dimensions,
            locations=locations,
            rotys=rotys,
            trans_mat=trans_mat,
            K=K,
            reg_mask=reg_mask,
            flip_mask=flip_mask,
            bbox_size=bbox_size,
            c_offsets=c_offsets)

    def prepare_predictions(self, targets_variables, pred_regression):
        """decode model predictions
        """
        batch, channel = pred_regression.shape[0], pred_regression.shape[1]
        targets_proj_points = targets_variables["proj_points"]

        # obtain prediction from points of interests
        pred_regression_pois = select_point_of_interest(
            batch, targets_proj_points, pred_regression)
        pred_regression_pois = paddle.reshape(pred_regression_pois,
                                              (-1, channel))

        # FIXME: fix hard code here
        pred_depths_offset = pred_regression_pois[:, 0]
        pred_proj_offsets = pred_regression_pois[:, 1:3]
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]
        pred_orientation = pred_regression_pois[:, 6:8]
        pred_bboxsize = pred_regression_pois[:, 8:10]

        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)
        pred_locations = self.smoke_coder.decode_location(
            targets_proj_points, pred_proj_offsets, pred_depths,
            targets_variables["K"], targets_variables["trans_mat"])

        pred_dimensions = self.smoke_coder.decode_dimension(
            targets_variables["cls_ids"],
            pred_dimensions_offsets,
        )
        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys = self.smoke_coder.decode_orientation(
            pred_orientation, targets_variables["locations"],
            targets_variables["flip_mask"])

        if self.reg_loss == "DisL1":
            pred_box3d_rotys = self.smoke_coder.encode_box3d(
                pred_rotys, targets_variables["dimensions"],
                targets_variables["locations"])

            pred_box3d_dims = self.smoke_coder.encode_box3d(
                targets_variables["rotys"], pred_dimensions,
                targets_variables["locations"])
            pred_box3d_locs = self.smoke_coder.encode_box3d(
                targets_variables["rotys"], targets_variables["dimensions"],
                pred_locations)

            return dict(
                ori=pred_box3d_rotys,
                dim=pred_box3d_dims,
                loc=pred_box3d_locs,
                bbox=pred_bboxsize,
            )

        elif self.reg_loss == "L1":
            pred_box_3d = self.smoke_coder.encode_box3d(
                pred_rotys, pred_dimensions, pred_locations)
            return pred_box_3d

    def __call__(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]

        targets_heatmap, targets_regression, targets_variables \
            = self.prepare_targets(targets)

        predict_boxes3d = self.prepare_predictions(targets_variables,
                                                   pred_regression)

        hm_loss = self.cls_loss(pred_heatmap,
                                targets_heatmap) * self.loss_weight[0]

        targets_regression = paddle.reshape(
            targets_regression,
            (-1, targets_regression.shape[2], targets_regression.shape[3]))

        reg_mask = targets_variables["reg_mask"].astype("float32").flatten()
        reg_mask = paddle.reshape(reg_mask, (-1, 1, 1))
        reg_mask = reg_mask.expand_as(targets_regression)

        if self.reg_loss == "DisL1":
            reg_loss_ori = F.l1_loss(
                predict_boxes3d["ori"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            reg_loss_dim = F.l1_loss(
                predict_boxes3d["dim"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            reg_loss_loc = F.l1_loss(
                predict_boxes3d["loc"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            reg_loss_size = F.l1_loss(
                predict_boxes3d["bbox"],
                paddle.reshape(targets_variables["bbox_size"],
                               (-1, targets_variables["bbox_size"].shape[-1])),
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            losses = dict(
                hm_loss=hm_loss,
                reg_loss=reg_loss_ori + reg_loss_dim + reg_loss_loc,
                size_loss=reg_loss_size)

            return hm_loss + reg_loss_ori + reg_loss_dim + reg_loss_loc + reg_loss_size
