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
import os
from os import path as osp
import cv2
import copy
import numpy as np
from PIL import Image

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle3d.apis import manager
from paddle3d.models.base import BaseMultiViewModel
from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample, SampleMeta
from paddle3d.utils import dtype2float32
from .petr3d import GridMask, bbox3d2result


def IOU(intputs, targets):
    numerator = 2 * (intputs * targets).sum(axis=1)
    denominator = intputs.sum(axis=1) + targets.sum(axis=1)
    loss = (numerator + 0.01) / (denominator + 0.01)
    return loss


@manager.MODELS.add_component
class Petr3D_seg(BaseMultiViewModel):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 backbone=None,
                 neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_recompute=False,
                 show=False,
                 box_with_velocity=False):
        num_cameras = 12 if pts_bbox_head.with_time else 6
        super(Petr3D_seg, self).__init__(
            box_with_velocity=box_with_velocity,
            num_cameras=num_cameras,
            need_timestamp=pts_bbox_head.with_time)
        self.pts_bbox_head = pts_bbox_head
        self.backbone = backbone
        self.neck = neck
        self.use_grid_mask = use_grid_mask
        self.use_recompute = use_recompute
        self.show = show

        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        self.init_weight()

    def init_weight(self, bias_lr_factor=0.1):
        for _, param in self.backbone.named_parameters():
            param.optimize_attr['learning_rate'] = bias_lr_factor

        self.pts_bbox_head.init_weights()

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if isinstance(img, list):
            img = paddle.stack(img, axis=0)

        B = img.shape[0]
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            if not getattr(self, 'in_export_mode', False):
                for img_meta in img_metas:
                    img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.shape[0] == 1 and img.shape[1] != 1:
                    if getattr(self, 'in_export_mode', False):
                        img = img.squeeze()
                    else:
                        img.squeeze_()
                else:
                    B, N, C, H, W = img.shape
                    img = img.reshape([B * N, C, H, W])
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            img_feats = self.neck(img_feats)
        else:
            return None

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.shape
            img_feats_reshaped.append(
                img_feat.reshape([B, int(BN / B), C, H, W]))

        return img_feats_reshaped

    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          maps,
                          img_metas,
                          gt_bboxes_ignore=None):
        """
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, maps]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    def train_forward(self,
                      samples=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      maps=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """
        """
        self.backbone.train()

        if samples is not None:
            img_metas = samples['meta']
            img = samples['img']
            maps = samples['maps']
            gt_labels_3d = samples['gt_labels_3d']
            gt_bboxes_3d = samples['gt_bboxes_3d']

        if hasattr(self, 'amp_cfg_'):
            with paddle.amp.auto_cast(**self.amp_cfg_):
                img_feats = self.extract_feat(img=img, img_metas=img_metas)
            img_feats = dtype2float32(img_feats)
        else:
            img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, maps, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        return dict(loss=losses)

    def test_forward(self, samples, gt_map=None, maps=None, img=None, **kwargs):
        img_metas = samples['meta']
        img = samples['img']
        maps = samples['maps']
        gt_map = samples['gt_map']

        img = [img] if img is None else img

        results = self.simple_test(img_metas, gt_map, img, maps, **kwargs)
        return dict(preds=results)

    def simple_test_pts(self,
                        x,
                        img_metas,
                        gt_map,
                        maps,
                        rescale=False,
                        output_dir='./output/petr_seg_imgs'):
        """Test function of point cloud branch."""

        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        with paddle.no_grad():

            lane_preds = outs['all_lane_preds'][5].squeeze(0)  #[B,N,H,W]
            n, w = lane_preds.shape

            pred_maps = lane_preds.reshape([256, 3, 16, 16])

            f_lane = pred_maps.reshape([16, 16, 3, 16, 16]).transpose(
                [2, 0, 3, 1, 4]).reshape([3, 256, 256])
            # f_lane = rearrange(pred_maps, '(h w) c h1 w2 -> c (h h1) (w w2)', h=16, w=16)
            f_lane = F.sigmoid(f_lane)
            f_lane = paddle.where(f_lane >= 0.5,
                                  paddle.ones(f_lane.shape, dtype='int32'),
                                  paddle.zeros(f_lane.shape, dtype='int32'))
            f_lane_show = copy.deepcopy(f_lane)
            gt_map_show = copy.deepcopy(gt_map[0])

            f_lane = f_lane.reshape([3, -1])
            gt_map = gt_map[0].reshape([3, -1])

            ret_iou = IOU(f_lane, gt_map)
            if self.show:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                pres = f_lane_show.numpy()
                pre = np.zeros([256, 256, 3])
                pre += 255
                label = [[71, 130, 255], [255, 255, 0], [255, 144, 30]]
                pre[..., 0][pres[0] > 0.5] = label[0][0]
                pre[..., 1][pres[0] > 0.23] = label[0][1]
                pre[..., 2][pres[0] > 0.56] = label[0][2]
                pre[..., 0][pres[2] > 0.5] = label[2][0]
                pre[..., 1][pres[2] > 0.23] = label[2][1]
                pre[..., 2][pres[2] > 0.56] = label[2][2]
                pre[..., 0][pres[1] > 0.5] = label[1][0]
                pre[..., 1][pres[1] > 0.23] = label[1][1]
                pre[..., 2][pres[1] > 0.56] = label[1][2]
                cv2.imwrite(
                    output_dir + '/' + img_metas[0]['sample_idx'] + '_pre' +
                    '.png', pre.astype(np.uint8))
                pres = gt_map_show.numpy()
                pre = np.zeros([256, 256, 3])
                pre += 255
                pre[..., 0][pres[0] > 0.5] = label[0][0]
                pre[..., 1][pres[0] > 0.5] = label[0][1]
                pre[..., 2][pres[0] > 0.5] = label[0][2]
                pre[..., 0][pres[2] > 0.5] = label[2][0]
                pre[..., 1][pres[2] > 0.5] = label[2][1]
                pre[..., 2][pres[2] > 0.5] = label[2][2]
                pre[..., 0][pres[1] > 0.5] = label[1][0]
                pre[..., 1][pres[1] > 0.5] = label[1][1]
                pre[..., 2][pres[1] > 0.5] = label[1][2]
                cv2.imwrite(
                    output_dir + '/' + img_metas[0]['sample_idx'] + '_gt' +
                    '.png', pre.astype(np.uint8))

        return bbox_results, ret_iou

    def simple_test(self,
                    img_metas,
                    gt_map=None,
                    img=None,
                    maps=None,
                    rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts, ret_iou = self.simple_test_pts(
            img_feats, img_metas, gt_map, maps, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['ret_iou'] = ret_iou
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(paddle.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def export_forward(self, samples):
        img = samples['images']
        img_metas = {'img2lidars': samples['img2lidars']}
        time_stamp = samples.get('timestamps', None)

        img_metas['image_shape'] = img.shape[-2:]
        img_feats = self.extract_feat(img=img, img_metas=None)

        bbox_list = [dict() for i in range(len(img_metas))]
        outs = self.pts_bbox_head.export_forward(img_feats, img_metas,
                                                 time_stamp)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, None, rescale=True)
        return bbox_list

    @property
    def save_name(self):
        return "petrv2_seg_inference"
