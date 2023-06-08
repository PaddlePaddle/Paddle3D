# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from BEV-LaneDet (https://github.com/gigo-team/bev_lane_det)
# ------------------------------------------------------------------------

import os
import math
import tempfile
import numpy as np
import os.path as osp

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle3d.models.backbones.mm_resnet import resnet34
from paddle3d.models.losses.push_pull_loss import NDPushPullLoss, IoULoss
from paddle3d.models.layers.param_init import kaiming_normal_init, constant_init, \
                        init_weight, _no_grad_uniform_, _calculate_fan_in_and_fan_out
from paddle3d.apis import manager


def naive_init_module(mod):

    for m in mod.sublayers():
        if isinstance(m, nn.Conv2D):
            kaiming_normal_init(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                _no_grad_uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
            constant_init(m.weight, value=1)
            constant_init(m.bias, value=0)
    return mod


def pairwise_dist(A, B):
    A = A.unsqueeze(-2)
    B = B.unsqueeze(-3)

    return paddle.abs(A - B).sum(-1)


class InstanceEmbeddingOffsetYZ(nn.Layer):
    def __init__(self, ci, co=1):
        super(InstanceEmbeddingOffsetYZ, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2D(ci, 128, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, ci, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2D(ci, 128, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 64, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 1, 3, 1, 1, bias_attr=True))

        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2D(ci, 128, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 64, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 1, 3, 1, 1, bias_attr=True))

        self.m_z = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2D(ci, 128, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 64, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 1, 3, 1, 1, bias_attr=True))

        self.me_new = nn.Sequential(
            nn.Conv2D(ci, 128, 3, 1, 1, bias_attr=False), nn.BatchNorm2D(128),
            nn.ReLU(), nn.Conv2D(128, 64, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(64), nn.ReLU(),
            nn.Conv2D(64, co, 3, 1, 1, bias_attr=True))

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.m_z)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(
            feat), self.m_z(feat)


class InstanceEmbedding(nn.Layer):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding, self).__init__()
        self.neck = nn.Sequential(
            nn.Conv2D(ci, 128, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, ci, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            nn.Conv2D(ci, 128, 3, 1, 1, bias_attr=False), nn.BatchNorm2D(128),
            nn.ReLU(), nn.Conv2D(128, 64, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(64), nn.ReLU(),
            nn.Conv2D(64, 1, 3, 1, 1, bias_attr=True))

        self.me = nn.Sequential(
            nn.Conv2D(ci, 128, 3, 1, 1, bias_attr=False), nn.BatchNorm2D(128),
            nn.ReLU(), nn.Conv2D(128, 64, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(64), nn.ReLU(),
            nn.Conv2D(64, co, 3, 1, 1, bias_attr=True))
        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


class LaneHeadResidualInstanceWithOffsetZ(nn.Layer):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidualInstanceWithOffsetZ, self).__init__()

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2D(input_channel, 64, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(64),
                    nn.ReLU(),
                    nn.Dropout2D(p=0.2),
                    nn.Conv2D(64, 128, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(128),
                ),
                downsample=nn.Conv2D(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2D(128, 64, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(64),
                    nn.ReLU(),
                    nn.Dropout2D(p=0.2),
                    nn.Conv2D(64, 64, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(64),
                ),
                downsample=nn.Conv2D(128, 64, 1),
            ),
        )
        self.head = InstanceEmbeddingOffsetYZ(64, 2)

        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)


class LaneHeadResidualInstance(nn.Layer):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidualInstance, self).__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Residual(
                module=nn.Sequential(
                    nn.Conv2D(input_channel, 64, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(64),
                    nn.ReLU(),
                    nn.Dropout2D(p=0.2),
                    nn.Conv2D(64, 128, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(128),
                ),
                downsample=nn.Conv2D(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),
            Residual(
                module=nn.Sequential(
                    nn.Conv2D(128, 64, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(64),
                    nn.ReLU(),
                    nn.Dropout2D(p=0.2),
                    nn.Conv2D(64, 32, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(32),
                ),
                downsample=nn.Conv2D(128, 32, 1),
            ),
            nn.Upsample(size=output_size),
            Residual(
                module=nn.Sequential(
                    nn.Conv2D(32, 16, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(16),
                    nn.ReLU(),
                    nn.Dropout2D(p=0.2),
                    nn.Conv2D(16, 32, 3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(32),
                )),
        )

        self.head = InstanceEmbedding(32, 2)

        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)


class FCTransform_(nn.Layer):
    def __init__(self, image_featmap_size, space_featmap_size):
        super(FCTransform_, self).__init__()
        ic, ih, iw = image_featmap_size  # (256, 16, 16)
        sc, sh, sw = space_featmap_size  # (128, 16, 32)
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw), nn.ReLU(), nn.Linear(sh * sw, sh * sw),
            nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv2D(
                in_channels=ic,
                out_channels=sc,
                kernel_size=1 * 1,
                stride=1,
                bias_attr=False),
            nn.BatchNorm2D(sc),
            nn.ReLU(),
        )
        self.residual = Residual(
            module=nn.Sequential(
                nn.Conv2D(
                    in_channels=sc,
                    out_channels=sc,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias_attr=False),
                nn.BatchNorm2D(sc),
            ))

        self.apply(init_weight)

    def forward(self, x):
        x = x.reshape(
            list(x.shape[:2]) + [
                self.image_featmap_size[1] * self.image_featmap_size[2],
            ])
        bev_view = self.fc_transform(x)
        bev_view = bev_view.reshape(
            list(bev_view.shape[:2]) +
            [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        return bev_view


class Residual(nn.Layer):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

        self.apply(init_weight)

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


@manager.MODELS.add_component
class BEVLaneDet(nn.Layer):
    def __init__(self,
                 bev_shape,
                 output_2d_shape,
                 train=True,
                 pretrained_model_path='resnet34-remapped.pdparams'):
        super(BEVLaneDet, self).__init__()
        self.bb = nn.Sequential(
            *list(resnet34(pretrained=pretrained_model_path).children()))

        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2D(512, 1024, kernel_size=3, stride=2,
                              padding=1),  # S64
                    nn.BatchNorm2D(1024),
                    nn.ReLU(),
                    nn.Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2D(1024)),
                downsample=nn.Conv2D(
                    512, 1024, kernel_size=3, stride=2, padding=1),
            ))

        self.s32transformer = FCTransform_((512, 18, 32), (256, 25, 5))
        self.s64transformer = FCTransform_((1024, 9, 16), (256, 25, 5))
        self.lane_head = LaneHeadResidualInstanceWithOffsetZ(
            bev_shape, input_channel=512)
        self.is_train = train
        if self.is_train:
            self.lane_head_2d = LaneHeadResidualInstance(
                output_2d_shape, input_channel=512)

        tmp_dir = tempfile.TemporaryDirectory()
        self.np_save_path = osp.join(tmp_dir.name, 'np_save')
        self.res_save_path = osp.join(tmp_dir.name, 'result')
        os.makedirs(self.np_save_path, exist_ok=True)

        self.bce = nn.BCEWithLogitsLoss(pos_weight=paddle.to_tensor([10.0]))
        self.iou_loss = IoULoss()
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, samples, *args, **kwargs):
        if self.training:
            return self.train_forward(samples, *args, **kwargs)

        return self.test_forward(samples, *args, **kwargs)

    def train_forward(self, samples, *args, **kwargs):
        img = samples['img']

        gt_seg = samples['bev_gt_segment']
        gt_instance = samples['bev_gt_instance']
        gt_offset_y = samples['bev_gt_offset']
        gt_z = samples['bev_gt_z']
        image_gt_segment = samples['image_gt_segment']
        image_gt_instance = samples['image_gt_instance']

        img_s32 = self.bb(img)
        img_s64 = self.down(img_s32)
        bev_32 = self.s32transformer(img_s32)
        bev_64 = self.s64transformer(img_s64)
        bev = paddle.concat([bev_64, bev_32], axis=1)

        pred, emb, offset_y, z = self.lane_head(bev)
        pred_2d, emb_2d = self.lane_head_2d(img_s32)

        loss_seg = self.bce(pred, gt_seg) + self.iou_loss(
            F.sigmoid(pred), gt_seg)

        loss_emb = self.poopoo(emb, gt_instance)
        loss_offset = self.bce_loss(gt_seg * F.sigmoid(offset_y), gt_offset_y)
        loss_z = self.mse_loss(gt_seg * z, gt_z)
        loss_total = 3 * loss_seg + 0.5 * loss_emb
        loss_total = loss_total.unsqueeze(0)
        loss_offset = 60 * loss_offset.unsqueeze(0)
        loss_z = 30 * loss_z.unsqueeze(0)
        ## 2d
        loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(
            F.sigmoid(pred_2d), image_gt_segment)
        loss_emb_2d = self.poopoo(emb_2d, image_gt_instance)
        loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d
        loss_total_2d = loss_total_2d.unsqueeze(0)

        losses = dict()
        losses['loss_total_bev'] = loss_total.mean()
        losses['loss_total_2d'] = loss_total_2d.mean()
        losses['loss_offset'] = loss_offset.mean()
        losses['loss_z'] = loss_z.mean()

        return dict(loss=losses)

    def test_forward(self, samples, *args, **kwargs):
        img = samples['img']
        bn_name = samples['name_list']
        img_s32 = self.bb(img)
        img_s64 = self.down(img_s32)
        bev_32 = self.s32transformer(img_s32)
        bev_64 = self.s64transformer(img_s64)
        bev = paddle.concat([bev_64, bev_32], axis=1)

        if self.is_train:
            pred_ = self.lane_head(bev), self.lane_head_2d(img_s32)
        else:
            pred_ = self.lane_head(bev)

        pred_ = pred_[0]
        seg = pred_[0].detach().cpu()
        embedding = pred_[1].detach().cpu()
        offset_y = paddle.nn.functional.sigmoid(pred_[2]).detach().cpu()
        z_pred = pred_[3].detach().cpu()
        np_save_path = self.np_save_path
        if not os.path.exists(np_save_path):
            os.makedirs(np_save_path)

        for idx in range(seg.shape[0]):
            ms, me, moffset, z = seg[idx].unsqueeze(0).numpy(
            ), embedding[idx].unsqueeze(0).numpy(), offset_y[idx].unsqueeze(
                0).numpy(), z_pred[idx].unsqueeze(0).numpy()
            tmp_res_for_save = np.concatenate((ms, me, moffset, z), axis=1)
            save_path = os.path.join(
                np_save_path,
                bn_name[0][idx] + '__' + bn_name[1][idx].replace('json', 'np'))

            np.save(save_path, tmp_res_for_save)

        return dict(preds=[np_save_path])
