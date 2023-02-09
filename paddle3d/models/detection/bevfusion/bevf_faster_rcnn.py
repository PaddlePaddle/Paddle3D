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

# This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/main/mmdet3d/models/detectors/bevf_faster_rcnn.py

import os

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.detection.bevfusion.cam_stream_lss import LiftSplatShoot
from paddle3d.models.detection.bevfusion.mvx_faster_rcnn import MVXFasterRCNN
from paddle3d.models.layers.param_init import reset_parameters
from paddle3d.models.necks.fpnc import ConvModule
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import logger

__all__ = ['BEVFFasterRCNN']


class SE_Block(nn.Layer):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2D(1), nn.Conv2D(c, c, kernel_size=1, stride=1),
            nn.Sigmoid())
        self.init_weights()

    def forward(self, x):
        return x * self.att(x)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2D):
                reset_parameters(m)

        self.apply(_init_weights)


@manager.MODELS.add_component
class BEVFFasterRCNN(MVXFasterRCNN):
    """Multi-modality BEVFusion."""

    def __init__(self,
                 lss=False,
                 lc_fusion=False,
                 camera_stream=False,
                 camera_depth_range=[4.0, 45.0, 1.0],
                 img_depth_loss_weight=1.0,
                 img_depth_loss_method='kld',
                 grid=0.6,
                 num_views=6,
                 se=False,
                 final_dim=(900, 1600),
                 pc_range=[-50, -50, -5, 50, 50, 3],
                 downsample=4,
                 imc=256,
                 lic=384,
                 load_img_from=None,
                 load_cam_from=None,
                 load_lidar_from=None,
                 **kwargs):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.

        """
        super(BEVFFasterRCNN, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.lift = camera_stream
        self.se = se
        if camera_stream:
            self.lift_splat_shot_vis = LiftSplatShoot(
                lss=lss,
                grid=grid,
                inputC=imc,
                camC=64,
                pc_range=pc_range,
                final_dim=final_dim,
                downsample=downsample)
        if lc_fusion:
            if se:
                self.seblock = SE_Block(lic)
            self.reduc_conv = ConvModule(
                lic + imc,
                lic,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(
                    type='BatchNorm2D', epsilon=1e-3, momentum=1 - 0.01),
                act_cfg=dict(type='ReLU'))

        self.freeze_img = kwargs.get('freeze_img', False)
        self.init_weights(pretrained=kwargs.get('pretrained', None))
        self.freeze()

        # load img, cam, lidar stream pretrained weight
        if load_img_from is not None:
            logger.info("load img weight from {}".format(load_img_from))
            load_pretrained_model(self, load_img_from, verbose=False)
        if load_cam_from is not None:
            logger.info("load cam weight from {}".format(load_cam_from))
            load_pretrained_model(self, load_cam_from, verbose=False)
        if load_lidar_from is not None:
            logger.info("load lidar weight from {}".format(load_lidar_from))
            load_pretrained_model(self, load_lidar_from, verbose=False)

    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.trainable = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.trainable = False
            if self.lift:
                for param in self.lift_splat_shot_vis.parameters():
                    param.trainable = False

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_backbone:
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

        if self.lift:
            BN, C, H, W = img_feats[0].shape
            batch_size = BN // self.num_views
            img_feats_view = img_feats[0].reshape(
                [batch_size, self.num_views, C, H, W])
            rots = []
            trans = []
            for sample_idx in range(batch_size):
                rot_list = []
                trans_list = []
                for mat in img_metas[sample_idx]['lidar2img']:
                    mat = mat.astype('float32')
                    rot_list.append(mat.inverse()[:3, :3])
                    trans_list.append(mat.inverse()[:3, 3].reshape([-1]))
                rot_list = paddle.stack(rot_list, axis=0)
                trans_list = paddle.stack(trans_list, axis=0)
                rots.append(rot_list)
                trans.append(trans_list)
            rots = paddle.stack(rots)
            trans = paddle.stack(trans)
            lidar2img_rt = img_metas[sample_idx]['lidar2img']

            img_bev_feat, depth_dist = self.lift_splat_shot_vis(
                img_feats_view,
                rots,
                trans,
                lidar2img_rt=lidar2img_rt,
                img_metas=img_metas)
            if pts_feats is None:
                pts_feats = [img_bev_feat]
            else:
                if self.lc_fusion:
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(
                            img_bev_feat,
                            pts_feats[0].shape[2:],
                            mode='bilinear',
                            align_corners=True)
                    pts_feats = [
                        self.reduc_conv(
                            paddle.concat([img_bev_feat, pts_feats[0]], axis=1))
                    ]
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]
        return dict(
            img_feats=img_feats, pts_feats=pts_feats, depth_dist=depth_dist)

    def export_forward(self, points, img, img_metas):
        """
        Args:
            points: paddle.Tensor, [num_points, 4]
            img: paddle.Tensor, [1, 6, 3, 480, 800]
            img_meats: List[]
        """
        # only for bs=1 forward
        # img_metas and points should be a list
        img_metas = [img_metas]
        points = [points]
        bbox_list = self.simple_test(points, img_metas, img)
        return bbox_list

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        depth_dist = feature_dict['depth_dist']

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

    def forward_train(self,
                      sample=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        if sample is not None:
            img = sample['img']
            img_metas = sample['img_metas']
            gt_bboxes_3d = sample['gt_bboxes_3d']
            gt_labels_3d = sample['gt_labels_3d']
            points = sample.get('points', None)
            img_depth = sample.get('img_depth', None)

        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        depth_dist = feature_dict['depth_dist']

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
            if img_depth is not None:
                loss_depth = self.depth_dist_loss(
                    depth_dist,
                    img_depth,
                    loss_method=self.img_depth_loss_method,
                    img=img) * self.img_depth_loss_weight
                losses.update(img_depth_loss=loss_depth)
            losses.update(losses_img)
        return losses

    def depth_dist_loss(self,
                        predict_depth_dist,
                        gt_depth,
                        loss_method='kld',
                        img=None):
        # predict_depth_dist: B, N, D, H, W
        # gt_depth: B, N, H', W'
        B, N, D, H, W = predict_depth_dist.shape
        guassian_depth, min_depth = gt_depth[..., 1:], gt_depth[..., 0]
        mask = (min_depth >= self.camera_depth_range[0]) & (
            min_depth <= self.camera_depth_range[1])
        mask = mask.reshape([-1])
        guassian_depth = guassian_depth.reshape([-1, D])[mask]
        predict_depth_dist = predict_depth_dist.transpose(
            [0, 1, 3, 4, 2]).reshape([-1, D])[mask]
        if loss_method == 'kld':
            loss = F.kl_div(
                paddle.log(predict_depth_dist),
                guassian_depth,
                reduction='mean')
        elif loss_method == 'mse':
            loss = F.mse_loss(predict_depth_dist, guassian_depth)
        else:
            raise NotImplementedError
        return loss

    def export(self, save_dir, **kwargs):
        self.forward = self.export_forward
        self.pts_middle_encoder.export_model = True
        self.lift_splat_shot_vis = True
        img_spec = paddle.static.InputSpec(
            shape=[1 * 6, 3, 448, 800], dtype='float32', name='img')
        pts_spec = paddle.static.InputSpec(
            shape=[-1, 4], dtype='float32', name='pts')
        img_metas_spec = {
            'lidar2img': [
                paddle.static.InputSpec(
                    shape=[4, 4], dtype='float32', name='lidar2img')
                for i in range(6)
            ]
        }
        input_spec = [pts_spec, img_spec, img_metas_spec]

        save_path = os.path.join(save_dir, 'bevfusion')
        paddle.jit.to_static(self, input_spec=input_spec)
        paddle.jit.save(self, save_path)
        logger.info("Exported model is saved in {}".format(save_path))
