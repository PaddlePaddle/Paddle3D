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

# This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/main/mmdet3d/models/detectors/mvx_two_stage.py

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.geometries import BBoxes3D
from paddle3d.models.voxelizers import HardVoxelizer
from paddle3d.sample import Sample, SampleMeta


class MVXTwoStageDetector(nn.Layer):
    """Base class of Multi-modality"""

    def __init__(
            self,
            sync_bn=False,
            freeze_img=True,
            pts_voxel_layer=None,
            pts_voxel_encoder=None,
            pts_middle_encoder=None,
            pts_fusion_layer=None,
            img_backbone=None,
            pts_backbone=None,
            img_neck=None,
            pts_neck=None,
            pts_bbox_head=None,
            img_roi_head=None,
            img_rpn_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super(MVXTwoStageDetector, self).__init__()

        self.sync_bn = sync_bn
        self.freeze_img = freeze_img
        if pts_voxel_layer:
            self.pts_voxel_layer = HardVoxelizer(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = pts_voxel_encoder
        if pts_middle_encoder:
            self.pts_middle_encoder = pts_middle_encoder
        if pts_backbone:
            self.pts_backbone = pts_backbone
        if pts_fusion_layer:
            self.pts_fusion_layer = pts_fusion_layer
        if pts_neck:
            self.pts_neck = pts_neck
        if pts_bbox_head:
            self.pts_bbox_head = pts_bbox_head

        if img_backbone:
            self.img_backbone = img_backbone
        if img_neck is not None:
            self.img_neck = img_neck
        if img_rpn_head is not None:
            self.img_rpn_head = img_rpn_head
        if img_roi_head is not None:
            self.img_roi_head = img_roi_head

        self.init_weights()

    def init_weights(self, pretrained=None):
        """Initialize model weights."""

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
        if self.with_img_backbone:
            self.img_backbone.init_weights(pretrained=img_pretrained)
        if self.with_pts_backbone:
            self.pts_backbone.init_weights(pretrained=pts_pretrained)
        if self.with_img_neck:
            if isinstance(self.img_neck, nn.Sequential):
                for m in self.img_neck:
                    m.init_weights()
            else:
                self.img_neck.init_weights()

        if self.with_pts_voxel_encoder:
            self.pts_voxel_encoder.init_weights()
        if self.with_pts_neck:
            self.pts_neck.init_weights()
        if self.with_img_roi_head:
            self.img_roi_head.init_weights(img_pretrained)
        if self.with_img_rpn:
            self.img_rpn_head.init_weights()
        if self.with_pts_bbox:
            self.pts_bbox_head.init_weights()
        if self.with_pts_roi_head:
            self.pts_roi_head.init_weights()

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.trainable = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.trainable = False

    @property
    def with_pts_roi_head(self):
        """bool: Whether the detector has a roi head in pts branch."""
        return hasattr(self, 'pts_roi_head') and self.pts_roi_head is not None

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self, 'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self,
                       'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_pts_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(
            self, 'pts_voxel_encoder') and self.pts_voxel_encoder is not None

    @property
    def with_pts_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(
            self, 'pts_middle_encoder') and self.pts_middle_encoder is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.shape[0] == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.shape[0] > 1:
                B, N, C, H, W = img.shape
                img = img.reshape([B * N, C, H, W])
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas, gt_bboxes_3d=None):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, coors, num_points = self.pts_voxel_layer(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        return [x]

    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    def forward_train(self,
                      sample=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[paddle.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[paddle.Tensor], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[paddle.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[paddle.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[paddle.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (paddle.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[paddle.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        if sample is not None:
            img = sample.get('img', None)
            img_metas = sample['img_metas']
            gt_bboxes_3d = sample['gt_bboxes_3d']
            gt_labels_3d = sample['gt_labels_3d']
            points = sample.get('points', None)

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[paddle.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[paddle.Tensor]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[paddle.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_img_train(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          proposals=None,
                          **kwargs):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[paddle.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            img_metas (list[dict]): Meta information of images.
            gt_bboxes (list[paddle.Tensor]): Ground truth boxes of each image
                sample.
            gt_labels (list[paddle.Tensor]): Ground truth labels of boxes.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            proposals (list[paddle.Tensor], optional): Proposals of each sample.
                Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        losses = dict()
        # RPN forward and loss
        if self.with_img_rpn:
            rpn_outs = self.img_rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.img_rpn)
            rpn_losses = self.img_rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('img_rpn_proposal',
                                              self.test_cfg.img_rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # bbox head forward and loss
        if self.with_img_bbox:
            # bbox head forward and loss
            img_roi_losses = self.img_roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, **kwargs)
            losses.update(img_roi_losses)

        return losses

    def forward(self, sample, **kwargs):
        if self.training:
            loss_dict = self.forward_train(sample, **kwargs)
            return {'loss': loss_dict}
        else:
            preds = self.forward_test(sample, **kwargs)
            return preds

    def forward_test(self, sample, **kwargs):
        """
        """
        img = sample.get('img', None)
        points = sample.get('points', None)
        img_metas = sample['img_metas']
        results = self.simple_test(points, img_metas, img, **kwargs)
        return dict(preds=self._parse_results_to_sample(results, sample))

    def simple_test_pts(self, x, img_metas, rescale=True):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=True):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def _parse_results_to_sample(self, results: dict, sample: dict):
        num_samples = len(results)
        new_results = []
        for i in range(num_samples):
            data = Sample(None, sample["modality"][i])
            bboxes_3d = results[i]['pts_bbox']["boxes_3d"].numpy()
            labels = results[i]['pts_bbox']["labels_3d"].numpy()
            confidences = results[i]['pts_bbox']["scores_3d"].numpy()
            bottom_center = bboxes_3d[:, :3]
            gravity_center = np.zeros_like(bottom_center)
            gravity_center[:, :2] = bottom_center[:, :2]
            gravity_center[:, 2] = bottom_center[:, 2] + bboxes_3d[:, 5] * 0.5
            bboxes_3d[:, :3] = gravity_center
            data.bboxes_3d = BBoxes3D(bboxes_3d[:, 0:7])
            data.bboxes_3d.coordmode = 'Lidar'
            data.bboxes_3d.origin = [0.5, 0.5, 0.5]
            data.bboxes_3d.rot_axis = 2
            data.bboxes_3d.velocities = bboxes_3d[:, 7:9]
            data['bboxes_3d_numpy'] = bboxes_3d[:, 0:7]
            data['bboxes_3d_coordmode'] = 'Lidar'
            data['bboxes_3d_origin'] = [0.5, 0.5, 0.5]
            data['bboxes_3d_rot_axis'] = 2
            data['bboxes_3d_velocities'] = bboxes_3d[:, 7:9]
            data.labels = labels
            data.confidences = confidences
            data.meta = SampleMeta(id=sample["meta"][i]['id'])
            if "calibs" in sample:
                calib = [calibs.numpy()[i] for calibs in sample["calibs"]]
                data.calibs = calib
            new_results.append(data)
        return new_results

    def collate_fn(self, batch):
        sample = batch[0]
        collated_batch = {}
        collated_fields = [
            'img', 'points', 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d',
            'modality', 'meta', 'idx', 'img_depth'
        ]
        for k in list(sample.keys()):
            if k not in collated_fields:
                continue
            if k == 'img':
                collated_batch[k] = np.stack([elem[k] for elem in batch],
                                             axis=0)
            elif k == 'img_depth':
                collated_batch[k] = paddle.stack(
                    [paddle.stack(elem[k], axis=0) for elem in batch], axis=0)
            else:
                collated_batch[k] = [elem[k] for elem in batch]
        return collated_batch


def bbox3d2result(bboxes, scores, labels):
    """Convert detection results to a list of numpy arrays.
    """
    return dict(
        boxes_3d=bboxes.cpu(), scores_3d=scores.cpu(), labels_3d=labels.cpu())
