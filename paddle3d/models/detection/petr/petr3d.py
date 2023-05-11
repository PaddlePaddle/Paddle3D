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
import os
from os import path as osp

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from PIL import Image
from paddle.static import InputSpec

from paddle3d.apis import manager, apply_to_static
from paddle3d.models.base import BaseMultiViewModel
from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample, SampleMeta
from paddle3d.utils import dtype2float32
from paddle3d.models.backbones.vovnetcp import _OSA_layer


class GridMask(nn.Layer):
    def __init__(self,
                 use_h,
                 use_w,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=0,
                 prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  #+ 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.shape
        x = x.reshape([-1, h, w])
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 +
                    h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = paddle.to_tensor(mask).astype('float32')
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = paddle.to_tensor(
                2 * (np.random.rand(h, w) - 0.5)).astype('float32')
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.reshape([n, c, h, w])


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.
    """
    result_dict = dict(
        boxes_3d=bboxes.cpu(), scores_3d=scores.cpu(), labels_3d=labels.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict


@manager.MODELS.add_component
class Petr3D(BaseMultiViewModel):
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
                 us_ms=False,
                 multi_scale=None,
                 box_with_velocity: bool = False,
                 to_static=False):
        num_cameras = 12 if pts_bbox_head.with_time else 6
        super(Petr3D, self).__init__(
            box_with_velocity=box_with_velocity,
            num_cameras=num_cameras,
            need_timestamp=pts_bbox_head.with_time)
        self.pts_bbox_head = pts_bbox_head
        self.backbone = backbone
        self.neck = neck
        self.to_static = to_static
        if self.to_static:
            for transformerlayer in self.pts_bbox_head.transformer.decoder.layers:
                transformerlayer.use_recompute = False
            for backbonelayer in self.backbone.sublayers():
                if isinstance(backbonelayer, _OSA_layer):
                    backbonelayer.use_checkpoint = False
            specs_head = ([
                InputSpec([1, 12, 256, 20, 50]),
                InputSpec([1, 12, 256, 10, 25])
            ], None, [
                InputSpec([4, 4], dtype=paddle.float64) for i in range(12)
            ], InputSpec([12]))
            specs_neck = [[
                InputSpec([12, 768, 20, 50]),
                InputSpec([12, 1024, 10, 25])
            ]]
            specs_backbone = [InputSpec([12, 3, 320, 800])]
            if not self.pts_bbox_head.with_denoise:
                self.pts_bbox_head.to_static = True
                apply_to_static(
                    to_static, self.pts_bbox_head, image_shape=specs_head)
            apply_to_static(
                to_static, self.backbone, image_shape=specs_backbone)
            apply_to_static(to_static, self.neck, image_shape=specs_neck)
        self.use_grid_mask = use_grid_mask
        self.use_recompute = use_recompute
        self.us_ms = us_ms
        if self.us_ms:
            self.multi_scale = multi_scale

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
            if self.us_ms:
                ms_img = []
                img_feats = []
                for scale in self.multi_scale:
                    s_img = F.interpolate(
                        img,
                        scale_factor=scale,
                        mode='bilinear',
                        align_corners=True)
                    ms_img.append(ms_img)
                    img_feat = self.backbone(s_img)
                    if isinstance(img_feat, dict):
                        img_feat = list(img_feat.values())
                    img_feats.append(img_feat)
                if len(self.multi_scale) > 1:
                    for i, scale in enumerate(self.multi_scale):
                        img_feats[i] = self.neck(img_feats[i])
                    if len(self.multi_scale) == 2:
                        img_feats = [
                            paddle.concat((img_feats[1][-2],
                                           F.interpolate(
                                               img_feats[0][-2],
                                               scale_factor=self.multi_scale[1]
                                               / self.multi_scale[0],
                                               mode='bilinear',
                                               align_corners=True)), 1)
                        ]
                    if len(self.multi_scale) == 3:
                        img_feats = [
                            paddle.concat((img_feats[2][-2],
                                           F.interpolate(
                                               img_feats[0][-2],
                                               scale_factor=self.multi_scale[2]
                                               / self.multi_scale[0],
                                               mode='bilinear',
                                               align_corners=True),
                                           F.interpolate(
                                               img_feats[1][-2],
                                               scale_factor=self.multi_scale[2]
                                               / self.multi_scale[1],
                                               mode='bilinear',
                                               align_corners=True)), 1)
                        ]
                else:
                    img_feats = self.neck(img_feats[-1])
            else:
                if os.environ.get('FLAGS_opt_layout',
                                  'False').lower() == 'true':
                    img_nhwc = paddle.transpose(img, [0, 2, 3, 1])
                    img_feats = []
                    img_feats_nhwc = self.backbone(img_nhwc)
                    if isinstance(img_feats_nhwc, dict):
                        img_feats_nhwc = list(img_feats_nhwc.values())
                    for img_feat_nhwc in img_feats_nhwc:
                        img_feats.append(
                            paddle.transpose(img_feat_nhwc, [0, 3, 1, 2]))
                else:
                    img_feats = self.backbone(img)

                    if hasattr(self, 'amp_cfg_'):
                        img_feats = dtype2float32(img_feats)

                    if isinstance(img_feats, dict):
                        img_feats = list(img_feats.values())
                # breakpoint()
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
                          img_metas,
                          gt_bboxes_ignore=None):
        """
        """
        if self.to_static and not self.pts_bbox_head.with_denoise:
            timestamp = paddle.to_tensor(np.asarray(img_metas[0]['timestamp']))
            outs = self.pts_bbox_head(
                pts_feats,
                img_metas=None,
                lidar2img=img_metas[0]['lidar2img'],
                timestamp=timestamp)
        else:
            outs = self.pts_bbox_head(pts_feats, img_metas=img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
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
            gt_labels_3d = samples['gt_labels_3d']
            gt_bboxes_3d = samples['gt_bboxes_3d']

        if hasattr(self, 'amp_cfg_'):
            with paddle.amp.auto_cast(**self.amp_cfg_):
                img_feats = self.extract_feat(img=img, img_metas=img_metas)
            img_feats = dtype2float32(img_feats)
        else:
            img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)

        return dict(loss=losses)

    def test_forward(self, samples, img=None, **kwargs):
        img_metas = samples['meta']
        img = samples['img']

        img = [img] if img is None else img

        results = self.simple_test(img_metas, img, **kwargs)
        return dict(preds=self._parse_results_to_sample(results, samples))

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""

        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
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
        if self.pts_bbox_head.with_time:
            return "petrv2_inference"
        return "petr_inference"

    @property
    def apollo_deploy_name(self):
        if self.pts_bbox_head.with_time:
            return "PETR_V2"
        return "PETR_V1"
