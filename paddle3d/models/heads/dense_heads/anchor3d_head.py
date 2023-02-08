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

# This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/3f992837ad659f050df38d7b0978372425be16ff/mmdet3d/models/dense_heads/anchor3d_head.py

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.detection.bevfusion.utils import box3d_multiclass_nms
from paddle3d.models.layers import param_init

from .anchor_mixins import AnchorTrainMixin, limit_period, multi_apply
from .samplers import PseudoSampler
from .target_assigner import MaxIoUAssigner

__all__ = ['Anchor3DHead']


@manager.HEADS.add_component
class Anchor3DHead(nn.Layer, AnchorTrainMixin):
    """Anchor head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 use_direction_classifier=True,
                 anchor_generator=None,
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=None,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_dir=None,
                 use_sigmoid_cls=True,
                 bbox_assigner=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset

        # anchor generator
        self.anchor_generator = anchor_generator
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.anchor_generator.num_base_anchors
        # box coder
        self.bbox_coder = bbox_coder
        self.box_code_size = self.bbox_coder.code_size

        # loss function
        self.use_sigmoid_cls = use_sigmoid_cls
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_dir = loss_dir

        # init head layers
        self._init_layers()

        # target assigner
        self.sampling = False
        self.bbox_sampler = PseudoSampler()
        self.bbox_assigner = MaxIoUAssigner(
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1)

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2D(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2D(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2D(self.feat_channels,
                                          self.num_anchors * 2, 1)

    def init_weights(self):
        """Initialize the weights of head."""
        # custom init
        bias_cls = param_init.init_bias_by_prob(0.01)
        param_init.normal_init(self.conv_cls.weight, std=0.01)
        if self.conv_cls.bias is not None:
            param_init.constant_init(self.conv_cls.bias, value=bias_cls)

        param_init.normal_init(self.conv_reg.weight, std=0.01)
        if self.conv_reg.bias is not None:
            param_init.constant_init(self.conv_reg.bias, value=0)

        # default init for self.conv_dir_cls
        if self.use_direction_classifier:
            param_init.reset_parameters(self.conv_dir_cls)

    def forward_single(self, x):
        """Forward function on a single-scale feature map.

        Args:
            x (paddle.Tensor): Input features.

        Returns:
            tuple[paddle.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[paddle.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[paddle.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, input_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.
            device (str): device of current module.

        Returns:
            list[list[paddle.Tensor]]: Anchors of each image, valid flags \
                of each image.
        """
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list

    def loss_single(self, cls_score, bbox_pred, dir_cls_preds, labels,
                    label_weights, bbox_targets, bbox_weights, dir_targets,
                    dir_weights, num_total_samples):
        """Calculate loss of Single-level results.

        Args:
            cls_score (paddle.Tensor): Class score in single-level.
            bbox_pred (paddle.Tensor): Bbox prediction in single-level.
            dir_cls_preds (paddle.Tensor): Predictions of direction class
                in single-level.
            labels (paddle.Tensor): Labels of class.
            label_weights (paddle.Tensor): Weights of class loss.
            bbox_targets (paddle.Tensor): Targets of bbox predictions.
            bbox_weights (paddle.Tensor): Weights of bbox loss.
            dir_targets (paddle.Tensor): Targets of direction predictions.
            dir_weights (paddle.Tensor): Weights of direction loss.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[paddle.Tensor]: Losses of class, bbox \
                and direction, respectively.
        """
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape([-1])
        label_weights = label_weights.reshape([-1])
        cls_score = cls_score.transpose([0, 2, 3,
                                         1]).reshape([-1, self.num_classes])
        assert labels.max().item() <= self.num_classes
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_pred = bbox_pred.transpose([0, 2, 3,
                                         1]).reshape([-1, self.box_code_size])
        bbox_targets = bbox_targets.reshape([-1, self.box_code_size])
        bbox_weights = bbox_weights.reshape([-1, self.box_code_size])

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) &
                    (labels < bg_class_ind)).nonzero(as_tuple=False)
        if pos_inds.numel() == 0:
            pos_inds = pos_inds.squeeze()
        else:
            pos_inds = pos_inds.reshape([-1])
        num_pos = len(pos_inds)

        pos_bbox_pred = paddle.gather(bbox_pred, pos_inds)
        pos_bbox_targets = paddle.gather(bbox_targets, pos_inds)
        pos_bbox_weights = paddle.gather(bbox_weights, pos_inds)

        # dir loss
        if self.use_direction_classifier:
            dir_cls_preds = dir_cls_preds.transpose([0, 2, 3,
                                                     1]).reshape([-1, 2])
            dir_targets = dir_targets.reshape([-1])
            dir_weights = dir_weights.reshape([-1])
            pos_dir_cls_preds = paddle.gather(dir_cls_preds, pos_inds)
            pos_dir_targets = paddle.gather(dir_targets, pos_inds)
            pos_dir_weights = paddle.gather(dir_weights, pos_inds)

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * paddle.to_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=num_total_samples)
        else:
            loss_bbox = 0  # pos_bbox_pred.sum()
            if self.use_direction_classifier:
                loss_dir = 0  # pos_dir_cls_preds.sum()

        return loss_cls, loss_bbox, loss_dir

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (paddle.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (paddle.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[paddle.Tensor]: ``boxes1`` and ``boxes2`` whose 7th \
                dimensions are changed.
        """
        rad_pred_encoding = paddle.sin(boxes1[..., 6:7]) * paddle.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = paddle.cos(boxes1[..., 6:7]) * paddle.sin(
            boxes2[..., 6:7])
        boxes1 = paddle.concat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], axis=-1)
        boxes2 = paddle.concat(
            [boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]], axis=-1)
        return boxes1, boxes2

    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             gt_bboxes,
             gt_labels,
             input_metas,
             gt_bboxes_ignore=None):
        """Calculate losses.

        Args:
            cls_scores (list[paddle.Tensor]): Multi-level class scores.
            bbox_preds (list[paddle.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[paddle.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Gt bboxes
                of each sample.
            gt_labels (list[paddle.Tensor]): Gt labels of each sample.
            input_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[paddle.Tensor]): Specify
                which bounding.

        Returns:
            dict[str, list[paddle.Tensor]]: Classification, bbox, and \
                direction losses of each level.

                - loss_cls (list[paddle.Tensor]): Classification losses.
                - loss_bbox (list[paddle.Tensor]): Box regression losses.
                - loss_dir (list[paddle.Tensor]): Direction classification \
                    losses.
        """
        featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        anchor_list = self.get_anchors(featmap_sizes, input_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            gt_bboxes,
            input_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos + num_total_neg
                             if self.sampling else num_total_pos)

        losses_cls, losses_bbox, losses_dir = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            dir_targets_list,
            dir_weights_list,
            num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   input_metas,
                   cfg=None,
                   rescale=False):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[paddle.Tensor]): Multi-level class scores.
            bbox_preds (list[paddle.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[paddle.Tensor]): Multi-level direction
                class predictions.
            input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[paddle.Tensor]): Whether th rescale bbox.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes)
        mlvl_anchors = [
            anchor.reshape([-1, self.box_code_size]) for anchor in mlvl_anchors
        ]

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id].detach() for i in range(num_levels)
            ]

            input_meta = input_metas[img_id]
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               dir_cls_pred_list, mlvl_anchors,
                                               input_meta, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          cfg=None,
                          rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (paddle.Tensor): Class score in single batch.
            bbox_preds (paddle.Tensor): Bbox prediction in single batch.
            dir_cls_preds (paddle.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[paddle.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[paddle.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (paddle.Tensor): Class score of each bbox.
                - labels (paddle.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]
            assert cls_score.shape[-2:] == dir_cls_pred.shape[-2:]
            dir_cls_pred = dir_cls_pred.transpose([1, 2, 0]).reshape([-1, 2])
            dir_cls_score = paddle.argmax(dir_cls_pred, axis=-1)

            cls_score = cls_score.transpose([1, 2,
                                             0]).reshape([-1, self.num_classes])
            if self.use_sigmoid_cls:
                scores = F.sigmoid(cls_score)
            else:
                scores = F.softmax(cls_score, axis=-1)
            bbox_pred = bbox_pred.transpose([1, 2, 0]).reshape(
                [-1, self.box_code_size])

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores = paddle.max(scores, axis=1)
                else:
                    max_scores = paddle.max(scores[:, :-1], axis=1)
                _, topk_inds = max_scores.topk(nms_pre)
                topk_inds = paddle.sort(topk_inds)
                anchors = paddle.gather(anchors, topk_inds, axis=0)
                bbox_pred = paddle.gather(bbox_pred, topk_inds, axis=0)
                scores = paddle.gather(scores, topk_inds, axis=0)
                dir_cls_score = paddle.gather(dir_cls_score, topk_inds, axis=0)

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = paddle.concat(mlvl_bboxes)
        # use 7-dimensions bboxes for nms,
        mlvl_bboxes_for_nms = mlvl_bboxes[:, :7].clone()
        mlvl_bboxes_for_nms[:, -1] = -mlvl_bboxes_for_nms[:, -1]
        # mlvl_bboxes_bev = paddle.gather(mlvl_bboxes,
        #                                 paddle.to_tensor([0, 1, 3, 4, 6]),
        #                                 axis=1)
        # mlvl_bboxes_for_nms = xywhr2xyxyr(mlvl_bboxes_bev)
        mlvl_scores = paddle.concat(mlvl_scores)
        mlvl_dir_scores = paddle.concat(mlvl_dir_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = paddle.zeros([mlvl_scores.shape[0], 1])
            mlvl_scores = paddle.concat([mlvl_scores, padding], axis=1)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, cfg['max_num'],
                                       cfg, mlvl_dir_scores)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (dir_rot + self.dir_offset +
                              np.pi * dir_scores.astype(bboxes.dtype))
        return bboxes, scores, labels


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (paddle.Tensor): Rotated boxes in XYWHR format.

    Returns:
        paddle.Tensor: Converted boxes in XYXYR format.
    """
    boxes = paddle.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes
