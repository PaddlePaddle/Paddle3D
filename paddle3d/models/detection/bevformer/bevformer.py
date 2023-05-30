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

import collections
import copy
import os
from typing import Dict, List

import numpy as np
import paddle
import paddle.nn as nn

from paddle3d.apis import manager
from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample, SampleMeta
from paddle3d.utils import dtype2float32
from paddle3d.utils.grid import GridMask
from paddle3d.utils.logger import logger
from paddle3d.slim.quant import QAT


@manager.MODELS.add_component
class BEVFormer(nn.Layer):
    def __init__(self,
                 backbone,
                 neck,
                 pts_bbox_head,
                 use_grid_mask=False,
                 pretrained=None,
                 video_test_mode=False):
        super(BEVFormer, self).__init__()
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.backbone = backbone
        self.neck = neck
        self.pts_bbox_head = pts_bbox_head
        self.pretrained = pretrained
        self.video_test_mode = video_test_mode
        self._quant = False

    def is_quant_model(self) -> bool:
        return self._quant

    def build_slim_model(self, slim_config: str):
        """ Slim the model and update the cfg params
        """
        self._quant = True

        logger.info("Build QAT model.")
        self.qat = QAT(quant_config=slim_config)
        # slim the model
        self.qat(self)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        if not getattr(self, 'export_model', False):
            B = img.shape[0]
            if img is not None:
                if img.dim() == 5 and img.shape[0] == 1:
                    img.squeeze_()
                elif img.dim() == 5 and img.shape[0] > 1:
                    B, N, C, H, W = img.shape
                    img = img.reshape([B * N, C, H, W])
                if self.use_grid_mask:
                    img = self.grid_mask(img)

                data = {'image': img}
                img_feats = self.backbone(data)
                if isinstance(img_feats, dict):
                    img_feats = list(img_feats.values())
            else:
                return None
        else:
            B = 1
            if self.use_grid_mask:
                img = self.grid_mask(img)
            data = {'image': img}
            img_feats = self.backbone(data)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        img_feats = self.neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.shape
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.reshape(
                        [int(B / len_queue), len_queue,
                         int(BN / B), C, H, W]))
            else:
                img_feats_reshaped.append(
                    img_feat.reshape([B, int(BN / B), C, H, W]))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(
            img=img, img_metas=img_metas, len_queue=len_queue)
        return img_feats

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with paddle.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape([bs * len_queue, num_cams, C, H, W])
            img_feats_list = self.extract_feat(
                img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                if prev_bev is None:
                    prev_bev = paddle.zeros([
                        self.pts_bbox_head.bev_w * self.pts_bbox_head.bev_w, bs,
                        self.pts_bbox_head.transformer.embed_dims
                    ],
                                            dtype='float32')
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def forward(self, samples, **kwargs):
        """
        """
        if self.training:
            if hasattr(self, 'amp_cfg_'):
                self.pts_bbox_head.amp_cfg_ = self.amp_cfg_
                with paddle.amp.auto_cast(
                        **self.amp_cfg_, custom_black_list=['linspace']):
                    return self.forward_train(samples, **kwargs)
            else:
                return self.forward_train(samples, **kwargs)
        else:
            return self.forward_test(samples, **kwargs)

    def forward_train(
            self,
            samples,
            gt_labels=None,
            gt_bboxes=None,
            proposals=None,
            gt_bboxes_ignore=None,
            img_depth=None,
            img_mask=None,
    ):

        img_metas = samples['meta']
        img = samples['img']
        gt_labels_3d = samples['gt_labels_3d']
        gt_bboxes_3d = samples['gt_bboxes_3d']

        bs = img.shape[0]
        len_queue = img.shape[1]
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        for i in range(prev_img.shape[1]):
            for each in prev_img_metas:
                each[i]['can_bus'] = paddle.to_tensor(each[i]['can_bus'])
                each[i]['lidar2img'] = [
                    paddle.to_tensor(ee) for ee in each[i]['lidar2img']
                ]

        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        for each in img_metas:
            each[len_queue - 1]['can_bus'] = paddle.to_tensor(
                each[len_queue - 1]['can_bus'])
            each[len_queue - 1]['lidar2img'] = [
                paddle.to_tensor(ee) for ee in each[len_queue - 1]['lidar2img']
            ]
        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        if prev_bev is None:
            prev_bev = paddle.zeros([
                self.pts_bbox_head.bev_w * self.pts_bbox_head.bev_w, bs,
                self.pts_bbox_head.transformer.embed_dims
            ],
                                    dtype='float32')

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses_pts = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        return losses_pts

    def forward_test(self, samples, **kwargs):
        img_metas = samples['meta']
        img = samples['img']
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        if self.prev_frame_info['prev_bev'] is None:
            self.prev_frame_info['prev_bev'] = paddle.zeros([
                self.pts_bbox_head.bev_w * self.pts_bbox_head.bev_w,
                img.shape[0], self.pts_bbox_head.transformer.embed_dims
            ],
                                                            dtype='float32')

        new_prev_bev, bbox_results = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return dict(preds=self._parse_results_to_sample(bbox_results, samples))

    def simple_test_pts(self, x, img_metas, prev_bev, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list

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

    def export_forward(self, img, prev_bev, img_metas):
        img_metas = [img_metas]
        new_prev_bev, bbox_results = self.simple_test(
            img_metas, img, prev_bev=prev_bev)
        return new_prev_bev, bbox_results

    def export(self, save_dir: str, **kwargs):
        self.forward = self.export_forward
        self.export_model = True
        self.pts_bbox_head.transformer.export_model = True
        self.pts_bbox_head.transformer.encoder.export_model = True
        image_spec = paddle.static.InputSpec(
            shape=[6, 3, 480, 800], dtype="float32", name='image')
        pre_bev_spec = paddle.static.InputSpec(
            shape=[
                self.pts_bbox_head.bev_w * self.pts_bbox_head.bev_w, 1,
                self.pts_bbox_head.transformer.embed_dims
            ],
            dtype="float32",
            name='pre_bev')
        img_metas_spec = {
            "can_bus":
            paddle.static.InputSpec(
                shape=[18], dtype="float32", name='can_bus'),
            "lidar2img":
            paddle.static.InputSpec(
                shape=[-1, -1, 4, 4], dtype="float32", name='lidar2img'),
            "img_shape":
            paddle.static.InputSpec(
                shape=[6, 3], dtype="int32", name='img_shape'),
        }

        input_spec = [image_spec, pre_bev_spec, img_metas_spec]

        paddle.jit.to_static(self, input_spec=input_spec)
        if self.is_quant_model:
            self.qat.save_quantized_model(
                model=self,
                path=os.path.join(save_dir, "bevformer_inference"),
                input_spec=input_spec)
        else:
            paddle.jit.save(self, os.path.join(save_dir, "bevformer_inference"))
        logger.info("Exported model is saved in {}".format(
            os.path.join(save_dir, "bevformer_inference")))


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.
    """
    result_dict = dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)

    if attrs is not None:
        result_dict['attrs_3d'] = attrs

    return result_dict
