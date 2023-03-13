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
This code is based on https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/bbox_heads/center_head.py
Ths copyright of tianweiy/CenterPoint is as follows:
MIT License [see LICENSE for details].

Portions of https://github.com/tianweiy/CenterPoint/blob/master/det3d/models/bbox_heads/center_head.py are from
det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)
Ths copyright of det3d is as follows:
MIT License [see LICENSE for details].
"""

import copy
import logging
from collections import defaultdict

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddle3d.apis import manager
from paddle3d.geometries.bbox import circle_nms
from paddle3d.models.backbones.second_backbone import build_conv_layer
from paddle3d.models.layers.layer_libs import rotate_nms_pcdet
from paddle3d.models.losses import FastFocalLoss, RegLoss
from paddle3d.models.voxel_encoders.pillar_encoder import build_norm_layer
from paddle3d.ops import centerpoint_postprocess
from paddle3d.utils.logger import logger


class ConvModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm_cfg=dict(type='BatchNorm2D', eps=1e-05, momentum=0.1)):
        super(ConvModule, self).__init__()
        # build convolution layer
        self.conv = build_conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
            distribution="norm")

        # build normalization layers
        norm_channels = out_channels
        self.bn = build_norm_layer(norm_cfg, norm_channels)

        # build activation layer
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


class SeparateHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 norm_cfg=dict(type='BatchNorm2D', eps=1e-05, momentum=0.1),
                 **kwargs):
        super(SeparateHead, self).__init__()
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

        with paddle.no_grad():
            for head in self.heads:
                if head == 'hm':
                    self.__getattr__(head)[-1].bias[:] = self.init_bias

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (paddle.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: paddle.Tensor]: contains the following keys:

                -reg （paddle.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (paddle.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (paddle.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (paddle.Tensor): Rotation value with the \
                    shape of [B, 2, H, W].
                -vel (paddle.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -hm (paddle.Tensor): hm with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads.keys():
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@manager.MODELS.add_component
class CenterHead(nn.Layer):
    def __init__(
            self,
            in_channels=[
                128,
            ],
            tasks=[],
            weight=0.25,
            code_weights=[],
            common_heads=dict(),
            init_bias=-2.19,
            share_conv_channel=64,
            num_hm_conv=2,
            norm_cfg=dict(type='BatchNorm2D', eps=1e-05, momentum=0.1),
    ):
        super(CenterHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.box_n_dim = 9 if 'vel' in common_heads else 7
        self.with_velocity = True if 'vel' in common_heads else False
        self.code_weights = code_weights
        self.use_direction_classifier = False

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg)

        self.tasks = nn.LayerList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SeparateHead(
                    init_bias=init_bias,
                    final_kernel=3,
                    in_channels=share_conv_channel,
                    heads=heads,
                    num_cls=num_cls))

        logger.info("Finish CenterHead Initialization")

    def forward(self, x, *kwargs):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts, x

    def _sigmoid(self, x):
        y = paddle.clip(F.sigmoid(x), min=1e-4, max=1 - 1e-4)
        return y

    def loss(self, example, preds_dicts, test_cfg, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # hm focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], example['heat_map'][task_id],
                                example['center_idx'][task_id],
                                example['target_mask'][task_id],
                                example['target_label'][task_id])

            target_box = example['target_bbox'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict:
                preds_dict['target_bbox'] = paddle.concat(
                    (preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                     preds_dict['vel'], preds_dict['rot']),
                    axis=1)
            else:
                preds_dict['target_bbox'] = paddle.concat(
                    (preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                     preds_dict['rot']),
                    axis=1)
                index = paddle.to_tensor([
                    0, 1, 2, 3, 4, 5, target_box.shape[-1] - 2,
                    target_box.shape[-1] - 1
                ],
                                         dtype='int32')
                target_box = paddle.index_select(
                    target_box, index=index, axis=-1)
                #target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]] # remove vel target

            ret = {}

            # Regression loss for dimension, offset, height, rotation
            box_loss = self.crit_reg(preds_dict['target_bbox'],
                                     example['target_mask'][task_id],
                                     example['center_idx'][task_id], target_box)

            loc_loss = (box_loss * paddle.to_tensor(
                self.code_weights, dtype=box_loss.dtype)).sum()

            loss = hm_loss + self.weight * loc_loss

            ret.update({
                'loss':
                loss,
                'hm_loss':
                hm_loss,
                'loc_loss':
                loc_loss,
                'loc_loss_elem':
                box_loss,
                'num_positive':
                paddle.cast(example['target_mask'][task_id],
                            dtype='float32').sum()
            })

            rets.append(ret)
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)
        rets_merged['loss'] = sum(rets_merged['loss'])
        return rets_merged

    @paddle.no_grad()
    def predict_by_custom_op(self, example, preds_dicts, test_cfg, **kwargs):
        rets = []
        metas = []
        hm = []
        reg = []
        height = []
        dim = []
        vel = []
        rot = []
        num_classes = []
        flag = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            for j, num_class in enumerate(self.num_classes):
                num_classes.append(flag)
                flag += num_class
            hm.append(preds_dict['hm'])
            reg.append(preds_dict['reg'])
            height.append(preds_dict['height'])
            dim.append(preds_dict['dim'])
            if self.with_velocity:
                vel.append(preds_dict['vel'])
            else:
                vel.append(preds_dict['reg'])
            rot.append(preds_dict['rot'])

        bboxes, scores, labels = centerpoint_postprocess.centerpoint_postprocess(
            hm, reg, height, dim, vel, rot, test_cfg.voxel_size,
            test_cfg.point_cloud_range, test_cfg.post_center_limit_range,
            num_classes, test_cfg.down_ratio, test_cfg.score_threshold,
            test_cfg.nms.nms_iou_threshold, test_cfg.nms.nms_pre_max_size,
            test_cfg.nms.nms_post_max_size, self.with_velocity)

        if "meta" not in example or len(example["meta"]) == 0:
            meta_list = [None]
        else:
            meta_list = example["meta"]

        ret_list = [{
            'meta': meta_list[0],
            'box3d_lidar': bboxes,
            'label_preds': labels,
            'scores': scores
        }]

        return ret_list

    @paddle.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing
        """
        # get loss info
        rets = []
        metas = []

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = paddle.to_tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W to N H W C
            for key, val in preds_dict.items():
                preds_dict[key] = val.transpose(perm=[0, 2, 3, 1])

            batch_size = preds_dict['hm'].shape[0]

            if "meta" not in example or len(example["meta"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["meta"]

            batch_hm = F.sigmoid(preds_dict['hm'])

            batch_dim = paddle.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            batch_rot = paddle.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.shape

            batch_reg = batch_reg.reshape([batch, H * W, 2])
            batch_hei = batch_hei.reshape([batch, H * W, 1])

            batch_rot = batch_rot.reshape([batch, H * W, 1])
            batch_dim = batch_dim.reshape([batch, H * W, 3])
            batch_hm = batch_hm.reshape([batch, H * W, num_cls])

            ys, xs = paddle.meshgrid([paddle.arange(0, H), paddle.arange(0, W)])

            ys = ys.reshape([1, H, W]).tile(repeat_times=[batch, 1, 1]).astype(
                batch_hm.dtype)
            xs = xs.reshape([1, H, W]).tile(repeat_times=[batch, 1, 1]).astype(
                batch_hm.dtype)

            xs = xs.reshape([batch, -1, 1]) + batch_reg[:, :, 0:1]
            ys = ys.reshape([batch, -1, 1]) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.down_ratio * test_cfg.voxel_size[
                0] + test_cfg.point_cloud_range[0]
            ys = ys * test_cfg.down_ratio * test_cfg.voxel_size[
                1] + test_cfg.point_cloud_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']

                batch_vel = batch_vel.reshape([batch, H * W, 2])
                batch_box_preds = paddle.concat(
                    [xs, ys, batch_hei, batch_dim, batch_vel, batch_rot],
                    axis=2)
            else:
                batch_box_preds = paddle.concat(
                    [xs, ys, batch_hei, batch_dim, batch_rot], axis=2)

            metas.append(meta_list)

            if test_cfg.get('per_class_nms', False):
                pass
            else:
                rets.append(
                    self.post_processing(batch_box_preds, batch_hm, test_cfg,
                                         post_center_range, task_id))

        # Merge branches results
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = paddle.concat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = paddle.concat([ret[i][k] for ret in rets])
            ret['meta'] = metas[0][i]
            ret_list.append(ret)

        return ret_list

    def single_post_processing(self, box_preds, hm_preds, test_cfg,
                               post_center_range, task_id):
        scores = paddle.max(hm_preds, axis=-1)
        labels = paddle.argmax(hm_preds, axis=-1)

        score_mask = scores > test_cfg.score_threshold
        distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
            & (box_preds[..., :3] <= post_center_range[3:]).all(1)

        mask = distance_mask & score_mask
        box_preds = box_preds[mask]
        scores = scores[mask]
        labels = labels[mask]

        def box_empty(box_preds, scores, labels, box_n_dim):
            # zero-shape tensor here will raise a error,
            # so we replace it with a fake result
            # prediction_dict = {
            #     'box3d_lidar': box_preds,
            #     'scores': scores,
            #     'label_preds': labels
            # }
            prediction_dict = {
                'box3d_lidar': paddle.zeros([1, box_n_dim],
                                            dtype=box_preds.dtype),
                'scores': -paddle.ones([1], dtype=scores.dtype),
                'label_preds': paddle.zeros([1], dtype=labels.dtype),
            }
            return prediction_dict

        def box_not_empty(box_preds, scores, labels, test_cfg):
            index = paddle.to_tensor(
                [0, 1, 2, 3, 4, 5, box_preds.shape[-1] - 1], dtype='int32')
            boxes_for_nms = paddle.index_select(box_preds, index=index, axis=-1)
            #boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            if test_cfg.get('circular_nms', False):
                centers = boxes_for_nms[:, [0, 1]]
                boxes = paddle.concat(
                    [centers, scores.reshape([-1, 1])], axis=1)
                selected = _circle_nms(
                    boxes,
                    min_radius=test_cfg.min_radius[task_id],
                    post_max_size=test_cfg.nms.nms_post_max_size)
            else:
                selected = rotate_nms_pcdet(
                    boxes_for_nms,
                    scores,
                    thresh=test_cfg.nms.nms_iou_threshold,
                    pre_max_size=test_cfg.nms.nms_pre_max_size,
                    post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected].reshape(
                [-1, box_preds.shape[-1]])
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }
            return prediction_dict

        return paddle.static.nn.cond(
            paddle.logical_not(mask.any()), lambda: box_empty(
                box_preds, scores, labels, self.box_n_dim), lambda:
            box_not_empty(box_preds, scores, labels, test_cfg))

    '''
    def single_post_processing(self, box_preds, hm_preds, test_cfg,
                               post_center_range, task_id):
        scores = paddle.max(hm_preds, axis=-1)
        labels = paddle.argmax(hm_preds, axis=-1)

        score_mask = scores > test_cfg.score_threshold
        distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
            & (box_preds[..., :3] <= post_center_range[3:]).all(1)

        mask = distance_mask & score_mask
        box_preds = box_preds[mask]
        scores = scores[mask]
        labels = labels[mask]

        # if 0 in box_preds.shape:
        #     prediction_dict = {
        #         'box3d_lidar': box_preds,
        #         'scores': scores,
        #         'label_preds': labels
        #     }
        #     return prediction_dict

        index = paddle.to_tensor(
            [0, 1, 2, 3, 4, 5, box_preds.shape[-1] - 1], dtype='int32')
        boxes_for_nms = paddle.index_select(box_preds, index=index, axis=-1)
        #boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

        if test_cfg.get('circular_nms', False):
            centers = boxes_for_nms[:, [0, 1]]
            boxes = paddle.concat([centers, scores.reshape([-1, 1])], axis=1)
            selected = _circle_nms(
                boxes,
                min_radius=test_cfg.min_radius[task_id],
                post_max_size=test_cfg.nms.nms_post_max_size)
        else:
            selected = rotate_nms_pcdet(
                boxes_for_nms,
                scores,
                thresh=test_cfg.nms.nms_iou_threshold,
                pre_maxsize=test_cfg.nms.nms_pre_max_size,
                post_max_size=test_cfg.nms.nms_post_max_size)

        selected_boxes = box_preds[selected].reshape([-1, box_preds.shape[-1]])
        selected_scores = scores[selected]
        selected_labels = labels[selected]

        prediction_dict = {
            'box3d_lidar': selected_boxes,
            'scores': selected_scores,
            'label_preds': selected_labels
        }
        return prediction_dict
    '''

    def post_processing(self, batch_box_preds, batch_hm, test_cfg,
                        post_center_range, task_id):
        if not getattr(self, "in_export_mode", False):
            batch_size = len(batch_hm)
            prediction_dicts = []
            for i in range(batch_size):
                box_preds = batch_box_preds[i]
                hm_preds = batch_hm[i]
                prediction_dict = self.single_post_processing(
                    box_preds, hm_preds, test_cfg, post_center_range, task_id)
                prediction_dicts.append(prediction_dict)

            return prediction_dicts
        else:
            prediction_dict = self.single_post_processing(
                batch_box_preds[0], batch_hm[0], test_cfg, post_center_range,
                task_id)
            return [prediction_dict]


import numpy as np


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.numpy(),
                               thresh=min_radius))[:post_max_size]

    keep = paddle.to_tensor(keep, dtype='int32')

    return keep
