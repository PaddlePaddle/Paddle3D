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
import copy
import cv2
import uuid

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from PIL import Image

from paddle3d.apis import manager
from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample, SampleMeta
from paddle3d.utils import dtype2float32

from einops import rearrange


def IOU(intputs, targets):
    numerator = 2 * (intputs * targets).sum(axis=1)
    denominator = intputs.sum(axis=1) + targets.sum(axis=1)
    loss = (numerator + 0.01) / (denominator + 0.01)
    return loss


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
class Petr3D_seg(nn.Layer):
    """Petr3D_seg."""

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
                 use_recompute=False):
        super(Petr3D_seg, self).__init__()
        
        self.pts_bbox_head = pts_bbox_head
        self.backbone = backbone
        self.neck = neck
        self.use_grid_mask = use_grid_mask
        self.use_recompute = use_recompute

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
        print('img in extract_img_feat: ', type(img))
        if isinstance(img, list):
            img = paddle.stack(img, axis=0)

        B = img.shape[0]
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            if not (hasattr(self, 'export_model') and self.export_model):
                for img_meta in img_metas:
                    img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.shape[0] == 1 and img.shape[1] != 1:
                    if hasattr(self, 'export_model') and self.export_model:
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
        else:
            return None

        img_feats = self.neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.shape
            img_feats_reshaped.append(
                img_feat.reshape([B, int(BN / B), C, H, W]))

        return img_feats_reshaped

    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        print('img in extract_feat: ', type(img))
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

    def forward(self, samples, **kwargs):
        """
        """
        if self.training:
            self.backbone.train()
            return self.forward_train(samples, **kwargs)
        else:
            return self.forward_test(samples, **kwargs)

    def forward_train(self,
                      samples=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      maps=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """
        """

        if samples is not None:
            img_metas = samples['meta']
            img = samples['img']
            gt_labels_3d = samples['gt_labels_3d']
            gt_bboxes_3d = samples['gt_bboxes_3d']
            maps = samples['maps']

        if hasattr(self, 'amp_cfg_'):
            with paddle.amp.auto_cast(**self.amp_cfg_):
                img_feats = self.extract_feat(img=img, img_metas=img_metas)
            img_feats = dtype2float32(img_feats)
        else:
            img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        #losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, maps, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)

        return dict(loss=losses)

    def forward_test(self, samples, gt_map=None, maps=None, img=None, **kwargs):
        img_metas = samples['meta']
        img = samples['img']
        gt_map = samples['gt_map']
        maps = samples['maps']
        
        img = [img] if img is None else img

        results = self.simple_test(img_metas, gt_map, img, maps, **kwargs)
        return dict(preds=self._parse_results_to_sample(results, samples))

    def simple_test_pts(self, x, img_metas, gt_map, maps, rescale=False):
        """Test function of point cloud branch."""

        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        
        with paddle.no_grad():
            lane_preds=outs['all_lane_preds'][5].squeeze(0)    #[B,N,H,W]
            n,w = lane_preds.shape
            #pred_maps = lane_preds.reshape([256,3,16,16])
            pred_maps = lane_preds.reshape([1024,3,16,16])
            f_lane = rearrange(pred_maps.cpu().numpy(), '(h w) c h1 w2 -> c (h h1) (w w2)', h=32, w=32)
            f_lane = F.sigmoid(paddle.to_tensor(f_lane))
            f_lane[f_lane>=0.5] = 1
            f_lane[f_lane<0.5] = 0
            f_lane_show=copy.deepcopy(f_lane)
            gt_map_show=copy.deepcopy(gt_map[0])
            
            f_lane=f_lane.reshape([3,-1])
            gt_map=gt_map[0].reshape([3,-1])
            
            ret_iou=IOU(f_lane, gt_map).cpu()
            show_res=False
            if show_res:
                save_uuid = str(uuid.uuid1())
                pres = f_lane_show.cpu().numpy()
                pre = np.zeros([512, 512, 3])
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
                #save_pred_path = '/notebooks/paddle3D/Paddle3D_for_develop/visible/res_pre/{}.png'.format(save_uuid)
                #cv2.imwrite(save_pred_path, pre.astype(np.uint8))
                pres = gt_map_show[0]
                pre = paddle.zeros([512, 512, 3])
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
                #save_gt_path = '/notebooks/paddle3D/Paddle3D_for_develop/visible/res_gt/{}.png'.format(save_uuid)
                #cv2.imwrite(save_gt_path, pres.cpu().numpy().astype(np.uint8) * 200)
        return bbox_results, ret_iou

    def simple_test(self, img_metas, gt_map=None, img=None, maps=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        #bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        bbox_pts, ret_iou = self.simple_test_pts(img_feats, img_metas, gt_map, maps, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['ret_iou'] = ret_iou
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

    def export_forward(self, img, img_metas, time_stamp=None):
        img_metas['image_shape'] = img.shape[-2:]
        img_feats = self.extract_feat(img=img, img_metas=None)

        bbox_list = [dict() for i in range(len(img_metas))]
        self.pts_bbox_head.export_model = True
        outs = self.pts_bbox_head.export_forward(img_feats, img_metas,
                                                 time_stamp)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, None, rescale=True)
        return bbox_list

    def export(self, save_dir: str, **kwargs):
        self.forward = self.export_forward
        self.export_model = True

        num_cams = 12 if self.pts_bbox_head.with_time else 6
        image_spec = paddle.static.InputSpec(
            shape=[1, num_cams, 3, 320, 800], dtype="float32")
        img2lidars_spec = {
            "img2lidars":
            paddle.static.InputSpec(
                shape=[1, num_cams, 4, 4], name='img2lidars'),
        }

        input_spec = [image_spec, img2lidars_spec]

        model_name = "petr_inference"
        if self.pts_bbox_head.with_time:
            time_spec = paddle.static.InputSpec(
                shape=[1, num_cams], dtype="float32")
            input_spec.append(time_spec)
            model_name = "petrv2_inference"

        paddle.jit.to_static(self, input_spec=input_spec)

        paddle.jit.save(self, os.path.join(save_dir, model_name))
