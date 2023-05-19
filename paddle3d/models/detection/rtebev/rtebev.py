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
import os
import time
import copy
import pickle
import collections
import numpy as np
from typing import Dict, List
from doctest import OutputChecker

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle3d.apis import manager
from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample, SampleMeta
from paddle3d.models.layers import param_init
from paddle3d.ops import bev_pool_v2


@manager.MODELS.add_component
class BEVDetFormer(nn.Layer):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self,
                 img_view_transformer,
                 img_bev_encoder_backbone,
                 img_bev_encoder_neck,
                 start_temporal_epoch=None,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 aux_pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 use_depth=False,
                 use_resnetvd=False,
                 use_ms_depth=False,
                 **kwargs):
        super(BEVDetFormer, self).__init__(**kwargs)

        self.pts_bbox_head = pts_bbox_head
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.with_img_neck = self.img_neck is not None
        self.img_view_transformer = img_view_transformer
        self.img_bev_encoder_backbone = img_bev_encoder_backbone
        self.img_bev_encoder_neck = img_bev_encoder_neck
        self.use_resnetvd = use_resnetvd
        self.use_ms_depth = use_ms_depth

        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = pre_process

        self.align_after_view_transfromation = align_after_view_transfromation  # if self.training else True
        self.num_frame = num_adj + 1
        self.with_prev = with_prev
        self.start_temporal_epoch = start_temporal_epoch
        self.use_depth = use_depth
        self.aux_pts_bbox_head = aux_pts_bbox_head

    def shift_feature(self, input, trans, rots, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _ = trans[0].shape

        # generate grid
        xs = paddle.linspace(
            0, w - 1, w, dtype=input.dtype).reshape((1, w)).expand((h, w))
        ys = paddle.linspace(
            0, h - 1, h, dtype=input.dtype).reshape((h, 1)).expand((h, w))
        grid = paddle.stack((xs, ys, paddle.ones_like(xs)), -1)
        grid = grid.reshape((1, h, w, 3)).expand((n, h, w, 3)).reshape((n, h, w,
                                                                        3, 1))

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = paddle.zeros((n, 1, 4, 4), dtype=grid.dtype)
        c02l0[:, :, :3, :3] = rots[0][:, 0:1, :, :]
        c02l0[:, :, :3, 3] = trans[0][:, 0:1, :]
        c02l0[:, :, 3, 3] = 1

        # transformation from adjacent camera frame to current ego frame
        c12l0 = paddle.zeros((n, 1, 4, 4), dtype=grid.dtype)
        c12l0[:, :, :3, :3] = rots[1][:, 0:1, :, :]
        c12l0[:, :, :3, 3] = trans[1][:, 0:1, :]
        c12l0[:, :, 3, 3] = 1

        # add bev data augmentation
        bda_ = paddle.zeros((n, 1, 4, 4), dtype=grid.dtype)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = paddle.zeros((n, 1, 4, 4), dtype=grid.dtype)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(paddle.inverse(c12l0))[:, 0, :, :].reshape(
            (n, 1, 1, 4, 4))
        l02l1 = l02l1.index_select(
            paddle.to_tensor([0, 1, 3]), axis=3).index_select(
                paddle.to_tensor([0, 1, 3]), axis=4)

        feat2bev = paddle.zeros((3, 3), dtype=grid.dtype)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.reshape((1, 3, 3))
        tf = paddle.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = paddle.to_tensor([w - 1.0, h - 1.0],
                                            dtype=input.dtype)
        grid = grid[:, :, :, :2, 0] / normalize_factor.reshape(
            (1, 1, 1, 2)) * 2.0 - 1.0
        output = F.grid_sample(
            input, grid.cast(input.dtype), align_corners=True)

        return output

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.reshape((B * N, C, imH, imW))
        if self.use_resnetvd:
            x = self.img_backbone(dict(image=imgs))
        else:
            x = self.img_backbone(imgs)

        if self.use_ms_depth:
            x = self.img_neck(x)

        else:
            if self.with_img_neck:  # todo check
                x = self.img_neck(x)
                if type(x) in [list, tuple]:
                    x = x[0]
        return x

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran, bda,
                         mlp_input):

        x = self.image_encoder(img)  # backbone + neck

        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]

        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, rots_curr, trans_curr, intrins = inputs[:4]
        rots_prev, trans_prev, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
            post_trans, bda[0:1, ...])
        inputs_curr = (imgs, rots_curr[0:1, ...], trans_curr[0:1, ...], intrins,
                       post_rots, post_trans, bda[0:1, ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [trans_curr, trans_prev],
                               [rots_curr, rots_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = paddle.concat(bev_feat_list, axis=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].reshape((B, N, self.num_frame, 3, H, W))
        imgs = paddle.split(imgs, imgs.shape[2], axis=2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.reshape((B, self.num_frame, N, 3, 3)),
            trans.reshape((B, self.num_frame, N, 3)),
            intrins.reshape((B, self.num_frame, N, 3, 3)),
            post_rots.reshape((B, self.num_frame, N, 3, 3)),
            post_trans.reshape((B, self.num_frame, N, 3))
        ]
        extra = [paddle.split(t, t.shape[1], 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        return imgs, rots, trans, intrins, post_rots, post_trans, bda

    def extract_img_feat(self, img, img_metas, **kwargs):
        x = self.image_encoder(img[0])
        if self.use_depth:
            _, rot, tran, intrin, post_rot, post_tran, bda = img[:7]
            mlp_input = self.img_view_transformer.get_mlp_input(
                rot, tran, intrin, post_rot, post_tran, bda)

            x, depth = self.img_view_transformer(
                [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        else:
            x, depth = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    def forward_train(self,
                      samples,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[paddle.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
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
        points = None
        img_metas = samples['meta']
        img = samples['img_inputs']
        gt_depth = samples['gt_depth']
        gt_labels_3d = samples['gt_labels_3d']
        gt_bboxes_3d = samples['gt_bboxes_3d']
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        if self.use_depth:
            loss_depth = self.img_view_transformer.get_depth_loss(
                gt_depth, depth)
            losses = dict(loss_depth=loss_depth)
        else:
            losses = dict()

        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)
        if self.aux_pts_bbox_head is not None:
            aux_losses_pts = self.forward_aux_pts_train(img_feats, gt_bboxes_3d,
                                                        gt_labels_3d, img_metas,
                                                        gt_bboxes_ignore)
            losses.update(aux_losses_pts)
        return {"loss": losses}

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[paddle.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
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
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_aux_pts_train(self,
                              pts_feats,
                              gt_bboxes_3d,
                              gt_labels_3d,
                              img_metas,
                              gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[paddle.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[paddle.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.aux_pts_bbox_head(pts_feats[0])
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs[0]]
        losses = self.aux_pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward(self, samples, *args, **kwargs):
        if self.training:
            return self.forward_train(samples, *args, **kwargs)

        self.align_after_view_transfromation = True
        return self.forward_test(samples, *args, **kwargs)

    def forward_test(self, samples, **kwargs):
        """
        Args:
            points (list[paddle.Tensor]): the outer list indicates test-time
                augmentations and inner paddle.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[paddle.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                paddle.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        points = None
        img_metas = [samples['meta']]
        img_inputs = samples['img_inputs']

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(samples, **kwargs)

        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentation."""
        assert False

    def _parse_results_to_sample(self, results: dict, sample: dict):
        num_samples = len(results)
        new_results = []
        for i in range(num_samples):
            data = Sample(None, sample["modality"][i])
            bboxes_3d = results[i][0].numpy()
            labels = results[i][2].numpy()
            confidences = results[i][1].numpy()

            data.bboxes_3d = BBoxes3D(bboxes_3d[:, [0, 1, 2, 3, 4, 5, -1]])
            data.bboxes_3d.coordmode = 'Lidar'
            data.bboxes_3d.origin = [0.5, 0.5, 0.5]
            data.bboxes_3d.rot_axis = 2
            if bboxes_3d.shape[
                    -1] == 9:  # box has shape (9,) (cx, cy, cz, w, h, l, {angle, vx, vy})
                data.bboxes_3d.velocities = bboxes_3d[:, 6:8]
            data.labels = labels
            data.confidences = confidences
            data.meta = SampleMeta(
                id=sample["meta"]["id"][0])  # fix id = meta {id: ...}
            if "calibs" in sample:
                calib = [calibs.numpy()[i] for calibs in sample["calibs"]]
                data.calibs = calib

            # ======
            # add to data for pickle save
            # pickle can only save 1 sub attribute
            data.coordmode = 'lidar'
            data.origin = [0.5, 0.5, 0.5]
            data.rot_axis = 2
            data.velocities = bboxes_3d[:, 6:8]

            new_results.append(data)
        return new_results

    def simple_test(self, samples, img=None, rescale=False, **kwargs):

        points = None
        img_metas = [samples['meta']]
        img = samples['img_inputs']
        """Test function without augmentation."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return {"preds": [result_dict]}

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)  # single apply

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            self.bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def bbox3d2result(self, bboxes, scores, labels, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (paddle.Tensor): Bounding boxes with shape (N, 5).
            labels (paddle.Tensor): Labels with shape (N, ).
            scores (paddle.Tensor): Scores with shape (N, ).
            attrs (paddle.Tensor, optional): Attributes with shape (N, ).
                Defaults to None.

        Returns:
            dict[str, paddle.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (paddle.Tensor): 3D boxes.
                - scores (paddle.Tensor): Prediction scores.
                - labels_3d (paddle.Tensor): Box labels.
                - attrs_3d (paddle.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.numpy(),
            scores_3d=scores.numpy(),
            labels_3d=labels.numpy())

        if attrs is not None:
            result_dict['attrs_3d'] = attrs

        return result_dict

    def forward_dummy(self, samples, **kwargs):
        points = None  #samples['points']
        img_metas = [samples['meta']]
        img_inputs = samples['img_inputs']
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs[0], img_metas=img_metas, **kwargs)
        outs = self.pts_bbox_head(
            img_feats[0])  # single apply in paddle centerhead
        return outs

    def get_bev_pool_input(self, input):
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)

    def export_forward(
            self,
            img,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x)
        depth = F.softmax(x[:, :self.img_view_transformer.D], axis=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        feat = tran_feat
        n, d, h, w = depth.shape
        feat = feat.reshape([n, feat.shape[1], h, w])
        feat = feat.transpose([0, 2, 3, 1])

        output_height, output_width = 128, 128
        bev_feat_shape = (1, output_height, output_width, 80)  # (B, Z, Y, X, C)
        out = bev_pool_v2.bev_pool_v2(depth, feat, ranks_depth, ranks_feat,
                                      ranks_bev, interval_lengths,
                                      interval_starts, bev_feat_shape)

        x = out.transpose((0, 3, 1, 2))
        x = x.reshape([1, 80, 128, 128])
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head([bev_feat], None)
        outs = self.pts_bbox_head.get_bboxes(outs, None)
        return outs

    def export(self, save_dir: str, **kwargs):
        self.forward = self.export_forward
        self.export_model = True

        deploy_cnt = 0
        for layer in self.sublayers():
            if hasattr(layer, 'convert_to_deploy'):
                layer.convert_to_deploy()
                deploy_cnt += 1
        print('Convert {} layer to deploy'.format(deploy_cnt))
        image_spec = paddle.static.InputSpec(
            shape=[6, 3, 320, 800], dtype="float32")
        ranks_depth_spec = paddle.static.InputSpec(shape=[None], dtype="int32")
        ranks_feat_spec = paddle.static.InputSpec(shape=[None], dtype="int32")
        ranks_bev_spec = paddle.static.InputSpec(shape=[None], dtype="int32")
        interval_starts_spec = paddle.static.InputSpec(
            shape=[None], dtype="int32")
        interval_lengths_spec = paddle.static.InputSpec(
            shape=[None], dtype="int32")

        input_spec = [
            image_spec, ranks_depth_spec, ranks_feat_spec, ranks_bev_spec,
            interval_starts_spec, interval_lengths_spec
        ]

        model_name = "bevpool"

        paddle.jit.to_static(self, input_spec=input_spec)
        paddle.jit.save(self, os.path.join(save_dir, model_name))


@manager.MODELS.add_component
class RTEBev(nn.Layer):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self,
                 img_view_transformer,
                 img_bev_encoder_backbone,
                 img_bev_encoder_neck,
                 start_temporal_epoch=None,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 aux_pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 use_depth=False,
                 use_resnetvd=False,
                 use_ms_depth=False,
                 **kwargs):
        super(RTEBev, self).__init__(**kwargs)

        self.pts_bbox_head = pts_bbox_head
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.with_img_neck = self.img_neck is not None
        self.img_view_transformer = img_view_transformer  #builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = img_bev_encoder_backbone  #builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = img_bev_encoder_neck  #builder.build_neck(img_bev_encoder_neck)

        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = pre_process

        self.align_after_view_transfromation = align_after_view_transfromation  # if self.training else True
        self.num_frame = num_adj + 1
        self.with_prev = with_prev
        self.start_temporal_epoch = start_temporal_epoch
        self.use_depth = use_depth
        self.aux_pts_bbox_head = aux_pts_bbox_head
        self.use_resnetvd = use_resnetvd
        self.use_ms_depth = use_ms_depth

    def shift_feature(self, input, trans, rots, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _ = trans[0].shape

        # generate grid
        xs = paddle.linspace(
            0, w - 1, w, dtype=input.dtype).reshape((1, w)).expand((h, w))
        ys = paddle.linspace(
            0, h - 1, h, dtype=input.dtype).reshape((h, 1)).expand((h, w))
        grid = paddle.stack((xs, ys, paddle.ones_like(xs)), -1)
        grid = grid.reshape((1, h, w, 3)).expand((n, h, w, 3)).reshape((n, h, w,
                                                                        3, 1))

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = paddle.zeros((n, 1, 4, 4), dtype=grid.dtype)
        c02l0[:, :, :3, :3] = rots[0][:, 0:1, :, :]
        c02l0[:, :, :3, 3] = trans[0][:, 0:1, :]
        c02l0[:, :, 3, 3] = 1

        # transformation from adjacent camera frame to current ego frame
        c12l0 = paddle.zeros((n, 1, 4, 4), dtype=grid.dtype)
        c12l0[:, :, :3, :3] = rots[1][:, 0:1, :, :]
        c12l0[:, :, :3, 3] = trans[1][:, 0:1, :]
        c12l0[:, :, 3, 3] = 1

        # add bev data augmentation
        bda_ = paddle.zeros((n, 1, 4, 4), dtype=grid.dtype)
        bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = paddle.zeros((n, 1, 4, 4), dtype=grid.dtype)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(paddle.inverse(c12l0))[:, 0, :, :].reshape(
            (n, 1, 1, 4, 4))
        l02l1 = l02l1.index_select(
            paddle.to_tensor([0, 1, 3]), axis=3).index_select(
                paddle.to_tensor([0, 1, 3]), axis=4)

        feat2bev = paddle.zeros((3, 3), dtype=grid.dtype)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.reshape((1, 3, 3))
        tf = paddle.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = paddle.to_tensor([w - 1.0, h - 1.0],
                                            dtype=input.dtype)
        grid = grid[:, :, :, :2, 0] / normalize_factor.reshape(
            (1, 1, 1, 2)) * 2.0 - 1.0
        output = F.grid_sample(
            input, grid.cast(input.dtype), align_corners=True)

        return output

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.reshape((B * N, C, imH, imW))
        x = self.img_backbone(imgs)
        if self.use_ms_depth:
            x = self.img_neck(x)
        else:
            if self.with_img_neck:  # todo check
                x = self.img_neck(x)
                if type(x) in [list, tuple]:
                    x = x[0]
        return x

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran, bda,
                         mlp_input):

        x = self.image_encoder(img)  # backbone + neck

        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]

        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, rots_curr, trans_curr, intrins = inputs[:4]
        rots_prev, trans_prev, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots_curr[0:1, ...], trans_curr[0:1, ...], intrins, post_rots,
            post_trans, bda[0:1, ...])
        inputs_curr = (imgs, rots_curr[0:1, ...], trans_curr[0:1, ...], intrins,
                       post_rots, post_trans, bda[0:1, ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [trans_curr, trans_prev],
                               [rots_curr, rots_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = paddle.concat(bev_feat_list, axis=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].reshape((B, N, self.num_frame, 3, H, W))
        imgs = paddle.split(imgs, imgs.shape[2], axis=2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.reshape((B, self.num_frame, N, 3, 3)),
            trans.reshape((B, self.num_frame, N, 3)),
            intrins.reshape((B, self.num_frame, N, 3, 3)),
            post_rots.reshape((B, self.num_frame, N, 3, 3)),
            post_trans.reshape((B, self.num_frame, N, 3))
        ]
        extra = [paddle.split(t, t.shape[1], 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        return imgs, rots, trans, intrins, post_rots, post_trans, bda

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])

        imgs, rots, trans, intrins, post_rots, post_trans, bda = \
           self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only

        for img, rot, tran, intrin, post_rot, post_tran in zip(
                imgs, rots, trans, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    rot, tran = rots[0], trans[0]
                t1 = time.time()
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot, post_tran, bda,
                               mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with paddle.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = paddle.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False

        if pred_prev:
            assert self.align_after_view_transfromation
            assert rots[0].shape[0] == 1
            feat_prev = paddle.concat(bev_feat_list[1:], axis=0)

            trans_curr = trans[0].tile([self.num_frame - 1, 1, 1])
            rots_curr = rots[0].tile([self.num_frame - 1, 1, 1, 1])
            trans_prev = paddle.concat(trans[1:], axis=0)
            rots_prev = paddle.concat(rots[1:], axis=0)

            bda_curr = bda.tile([self.num_frame - 1, 1, 1])
            return feat_prev, [
                imgs[0], rots_curr, trans_curr, intrins[0], rots_prev,
                trans_prev, post_rots[0], post_trans[0], bda_curr
            ]

        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [trans[0], trans[adj_id]],
                                       [rots[0], rots[adj_id]],
                                       bda)
        bev_feat = paddle.concat(bev_feat_list, axis=1)

        x = self.bev_encoder(bev_feat)

        return [x], depth_list[0]

    def extract_img_feat_single(self, img, img_metas, **kwargs):
        x = self.image_encoder(img[0])
        if self.use_depth:
            _, rot, tran, intrin, post_rot, post_tran, bda = img[:7]
            mlp_input = self.img_view_transformer.get_mlp_input(
                rot, tran, intrin, post_rot, post_tran, bda)

            x, depth = self.img_view_transformer(
                [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        else:
            x, depth = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        if self.num_frame == 1:
            img_feats, depth = self.extract_img_feat_single(
                img, img_metas, **kwargs)
        else:
            img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)

        pts_feats = None

        return (img_feats, pts_feats, depth)

    def forward_train(self,
                      samples,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[paddle.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
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
        points = None
        img_metas = samples['meta']
        img = samples['img_inputs']
        gt_depth = samples['gt_depth']
        gt_labels_3d = samples['gt_labels_3d']
        gt_bboxes_3d = samples['gt_bboxes_3d']
        img_feats, _, depth = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        if self.use_depth:
            loss_depth = self.img_view_transformer.get_depth_loss(
                gt_depth, depth)
            losses = dict(loss_depth=loss_depth)
        else:
            losses = dict()

        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)

        if self.aux_pts_bbox_head is not None:
            aux_losses_pts = self.forward_aux_pts_train(img_feats, gt_bboxes_3d,
                                                        gt_labels_3d, img_metas,
                                                        gt_bboxes_ignore)
            losses.update(aux_losses_pts)

        return {"loss": losses}

    def forward_aux_pts_train(self,
                              pts_feats,
                              gt_bboxes_3d,
                              gt_labels_3d,
                              img_metas,
                              gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[paddle.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[paddle.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.aux_pts_bbox_head(pts_feats[0])  # single apply
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs[0]]
        losses = self.aux_pts_bbox_head.loss(*loss_inputs)
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
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[paddle.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[paddle.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)  # single apply
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward(self, samples, *args, **kwargs):

        if self.training:
            return self.forward_train(samples, *args, **kwargs)

        self.align_after_view_transfromation = True
        return self.forward_test(samples, *args, **kwargs)

    def forward_test(self, samples, **kwargs):
        """
        Args:
            points (list[paddle.Tensor]): the outer list indicates test-time
                augmentations and inner paddle.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[paddle.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                paddle.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        points = None
        img_metas = [samples['meta']]
        img_inputs = samples['img_inputs']

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(samples, **kwargs)

        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentation."""
        assert False

    def _parse_results_to_sample(self, results: dict, sample: dict):
        num_samples = len(results)
        new_results = []
        for i in range(num_samples):
            data = Sample(None, sample["modality"][i])

            bboxes_3d = results[i][0].numpy()
            labels = results[i][2].numpy()
            confidences = results[i][1].numpy()

            data.bboxes_3d = BBoxes3D(bboxes_3d[:, [0, 1, 2, 3, 4, 5, -1]])
            data.bboxes_3d.coordmode = 'Lidar'
            data.bboxes_3d.origin = [0.5, 0.5, 0.5]
            data.bboxes_3d.rot_axis = 2
            if bboxes_3d.shape[
                    -1] == 9:  # box has shape (9,) (cx, cy, cz, w, h, l, {angle, vx, vy})
                data.bboxes_3d.velocities = bboxes_3d[:, 6:8]
            data.labels = labels
            data.confidences = confidences
            data.meta = SampleMeta(
                id=sample["meta"]["id"][0])  # fix id = meta {id: ...}
            if "calibs" in sample:
                calib = [calibs.numpy()[i] for calibs in sample["calibs"]]
                data.calibs = calib

            # ======
            # add to data for pickle save
            # pickle can only save 1 sub attribute
            data.coordmode = 'lidar'
            data.origin = [0.5, 0.5, 0.5]
            data.rot_axis = 2
            data.velocities = bboxes_3d[:, 6:8]

            new_results.append(data)
        return new_results

    def simple_test(self, samples, img=None, rescale=False, **kwargs):

        points = None
        img_metas = [samples['meta']]
        img = samples['img_inputs']
        """Test function without augmentation."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return {"preds": [result_dict]}

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)  # single apply

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            self.bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results

    def bbox3d2result(self, bboxes, scores, labels, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (paddle.Tensor): Bounding boxes with shape (N, 5).
            labels (paddle.Tensor): Labels with shape (N, ).
            scores (paddle.Tensor): Scores with shape (N, ).
            attrs (paddle.Tensor, optional): Attributes with shape (N, ).
                Defaults to None.

        Returns:
            dict[str, paddle.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (paddle.Tensor): 3D boxes.
                - scores (paddle.Tensor): Prediction scores.
                - labels_3d (paddle.Tensor): Box labels.
                - attrs_3d (paddle.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.numpy(),
            scores_3d=scores.numpy(),
            labels_3d=labels.numpy())

        if attrs is not None:
            result_dict['attrs_3d'] = attrs

        return result_dict

    def get_bev_pool_input(self, input):
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)

    def mf_export_forward(self, img, feat_prev, mlp_input, ranks_depth,
                          ranks_feat, ranks_bev, interval_starts,
                          interval_lengths):

        self.align_after_view_transfromation = True
        x = self.image_encoder(img)
        if self.use_ms_depth:
            depth_digit, tran_feat = self.img_view_transformer.depth_net(
                x[0], x[1], x[2], mlp_input)
            depth = F.softmax(depth_digit, axis=1)
        else:
            B, N, C, H, W = x.shape
            x = x.reshape((B * N, C, H, W))
            x = self.img_view_transformer.depth_net(x, mlp_input)
            depth = F.softmax(x[:, :self.img_view_transformer.D], axis=1)
            tran_feat = x[:, self.img_view_transformer.D:(
                self.img_view_transformer.D +
                self.img_view_transformer.out_channels)]
        feat = tran_feat
        n, d, h, w = depth.shape
        feat = feat.reshape([n, feat.shape[1], h, w])
        feat = feat.transpose([0, 2, 3, 1])

        output_height, output_width = 128, 128
        bev_feat_shape = (1, output_height, output_width, 80)  # (B, Z, Y, X, C)
        out = bev_pool_v2.bev_pool_v2(depth, feat, ranks_depth, ranks_feat,
                                      ranks_bev, interval_lengths,
                                      interval_starts, bev_feat_shape)
        bev_feat = out.transpose((0, 3, 1, 2))
        bev_feat = bev_feat.reshape([1, 80, 128, 128])

        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]

        bev_feat_list = []
        bev_feat_list.append(bev_feat)
        bev_feat_list.append(feat_prev)
        bev_feat = paddle.concat(bev_feat_list, axis=1)
        bev_feat = self.bev_encoder(bev_feat)

        outs = self.pts_bbox_head([bev_feat], None)
        outs = self.pts_bbox_head.get_bboxes(outs, None, rescale=False)
        return outs

    def export_forward(self,
                       img,
                       ranks_depth,
                       ranks_feat,
                       ranks_bev,
                       interval_starts,
                       interval_lengths,
                       mlp_input=None):
        x = self.img_backbone(img)
        x = self.img_neck(x)

        if self.use_depth:
            if self.use_ms_depth:
                depth_digit, tran_feat = self.img_view_transformer.depth_net(
                    x[0], x[1], x[2], mlp_input)
                depth = F.softmax(depth_digit, axis=1)
            else:
                x = self.img_view_transformer.depth_net(x, mlp_input)
                depth = F.softmax(x[:, :self.img_view_transformer.D], axis=1)
                tran_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D +
                    self.img_view_transformer.out_channels)]
        else:
            x = self.img_view_transformer.depth_net(x)
            depth = F.softmax(x[:, :self.img_view_transformer.D], axis=1)
            tran_feat = x[:, self.img_view_transformer.D:(
                self.img_view_transformer.D +
                self.img_view_transformer.out_channels)]

        feat = tran_feat
        n, d, h, w = depth.shape
        feat = feat.reshape([n, feat.shape[1], h, w])
        feat = feat.transpose([0, 2, 3, 1])

        output_height, output_width = 128, 128
        bev_feat_shape = (1, output_height, output_width, 80)  # (B, Z, Y, X, C)
        out = bev_pool_v2.bev_pool_v2(depth, feat, ranks_depth, ranks_feat,
                                      ranks_bev, interval_lengths,
                                      interval_starts, bev_feat_shape)

        x = out.transpose((0, 3, 1, 2))
        x = x.reshape([1, 80, 128, 128])
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head([bev_feat], None)
        outs = self.pts_bbox_head.get_bboxes(outs, None)
        return outs

    def export(self, save_dir: str, **kwargs):
        if self.num_frame == 1:
            self.forward = self.export_forward
            self.export_model = True

            deploy_cnt = 0
            for layer in self.sublayers():
                if hasattr(layer, 'convert_to_deploy'):
                    layer.convert_to_deploy()
                    deploy_cnt += 1
            print('Convert {} layer to deploy'.format(deploy_cnt))
            if os.environ.get('FLAGS_onnx'):
                print('onnx export')
                image_spec = paddle.static.InputSpec(
                    shape=[6, 3, 256, 704], dtype="float32", name='image')
                ranks_depth_spec = paddle.static.InputSpec(
                    shape=[15010], dtype="int32", name='ranks_depth')
                ranks_feat_spec = paddle.static.InputSpec(
                    shape=[15010], dtype="int32", name='ranks_feat')
                ranks_bev_spec = paddle.static.InputSpec(
                    shape=[728985], dtype="int32", name='ranks_bev')
                interval_starts_spec = paddle.static.InputSpec(
                    shape=[728985], dtype="int32", name='interval_starts')
                interval_lengths_spec = paddle.static.InputSpec(
                    shape=[728985], dtype="int32", name='interval_lengths')

                input_spec = [
                    image_spec, ranks_depth_spec, ranks_feat_spec,
                    ranks_bev_spec, interval_starts_spec, interval_lengths_spec
                ]
                model_name = "rtebev/model"
                if self.use_depth:
                    mlp_input_spec = paddle.static.InputSpec(
                        shape=[1, 6, 27], dtype="float32", name='mlp_input')
                input_spec += [mlp_input_spec]
                paddle.onnx.export(
                    self,
                    os.path.join(save_dir, model_name),
                    input_spec=input_spec,
                    opset_version=13,
                    enable_onnx_checker=True,
                    custom_ops={"bev_pool_v2": "bev_pool_v2"})

            else:
                image_spec = paddle.static.InputSpec(
                    shape=[6, 3, 256, 704], dtype="float32", name='image')
                ranks_depth_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='ranks_depth')
                ranks_feat_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='ranks_feat')
                ranks_bev_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='ranks_bev')
                interval_starts_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='interval_starts')
                interval_lengths_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='interval_lengths')

                input_spec = [
                    image_spec, ranks_depth_spec, ranks_feat_spec,
                    ranks_bev_spec, interval_starts_spec, interval_lengths_spec
                ]

                model_name = "rtebev/model"
                if self.use_depth:
                    mlp_input_spec = paddle.static.InputSpec(
                        shape=[1, 6, 27], dtype="float32", name='mlp_input')
                    input_spec += [mlp_input_spec]

                paddle.jit.to_static(self, input_spec=input_spec)
                paddle.jit.save(self, os.path.join(save_dir, model_name))

        else:
            self.forward = self.mf_export_forward
            self.export_model = True

            if os.environ.get('FLAGS_onnx'):
                print('onnx export')
                image_spec = paddle.static.InputSpec(
                    shape=[1, 6, 3, 256, 704], dtype="float32", name='image')
                feat_prev_spec = paddle.static.InputSpec(
                    shape=[1, (self.num_frame - 1) * 80, 128, 128],
                    dtype="float32",
                    name="feat_prev")
                mlp_input_spec = paddle.static.InputSpec(
                    shape=[1, 6, 27], dtype="float32", name="mlp_input")
                ranks_depth_spec = paddle.static.InputSpec(
                    shape=[15010], dtype="int32", name='ranks_depth')
                ranks_feat_spec = paddle.static.InputSpec(
                    shape=[15010], dtype="int32", name='ranks_feat')
                ranks_bev_spec = paddle.static.InputSpec(
                    shape=[728985], dtype="int32", name='ranks_bev')
                interval_starts_spec = paddle.static.InputSpec(
                    shape=[728985], dtype="int32", name='interval_starts')
                interval_lengths_spec = paddle.static.InputSpec(
                    shape=[728985], dtype="int32", name='interval_lengths')
                input_spec = [
                    image_spec, feat_prev_spec, mlp_input_spec,
                    ranks_depth_spec, ranks_feat_spec, ranks_bev_spec,
                    interval_starts_spec, interval_lengths_spec
                ]

                model_name = "rtebev_mf/model"
                paddle.onnx.export(
                    self,
                    os.path.join(save_dir, model_name),
                    input_spec=input_spec,
                    opset_version=13,
                    enable_onnx_checker=True,
                    custom_ops={"bev_pool_v2": "bev_pool_v2"})
            else:
                image_spec = paddle.static.InputSpec(
                    shape=[1, 6, 3, 256, 704], dtype="float32", name='image')
                feat_prev_spec = paddle.static.InputSpec(
                    shape=[1, (self.num_frame - 1) * 80, 128, 128],
                    dtype="float32",
                    name="feat_prev")
                mlp_input_spec = paddle.static.InputSpec(
                    shape=[1, 6, 27], dtype="float32", name="mlp_input")
                ranks_depth_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='ranks_depth')
                ranks_feat_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='ranks_feat')
                ranks_bev_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='ranks_bev')
                interval_starts_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='interval_starts')
                interval_lengths_spec = paddle.static.InputSpec(
                    shape=[None], dtype="int32", name='interval_lengths')

                input_spec = [
                    image_spec, feat_prev_spec, mlp_input_spec,
                    ranks_depth_spec, ranks_feat_spec, ranks_bev_spec,
                    interval_starts_spec, interval_lengths_spec
                ]

                model_name = "rtebev_mf/model"

                paddle.jit.to_static(self, input_spec=input_spec)
                paddle.jit.save(self, os.path.join(save_dir, model_name))

    def forward_dummy(self, samples, **kwargs):
        points = None  #samples['points']
        img_metas = [samples['meta']]
        img_inputs = samples['img_inputs']
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs[0], img_metas=img_metas, **kwargs)
        outs = self.pts_bbox_head(
            img_feats[0])  # single apply in paddle centerhead
        return outs


class HoriConv(nn.Layer):
    def __init__(self, in_channels, mid_channels, out_channels, cat_dim=0):
        """HoriConv that reduce the image feature
            in height dimension and refine it.

        Args:
            in_channels (int): in_channels
            mid_channels (int): mid_channels
            out_channels (int): output channels
            cat_dim (int, optional): channels of position
                embedding. Defaults to 0.
        """
        super().__init__()

        self.merger = nn.Sequential(
            nn.Conv2D(
                in_channels + cat_dim,
                in_channels,
                kernel_size=1,
                bias_attr=True),
            nn.Sigmoid(),
            nn.Conv2D(in_channels, in_channels, kernel_size=1, bias_attr=True),
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv1D(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            nn.BatchNorm1D(mid_channels),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1D(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            nn.BatchNorm1D(mid_channels),
            nn.ReLU(),
            nn.Conv1D(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            nn.BatchNorm1D(mid_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1D(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            nn.BatchNorm1D(mid_channels),
            nn.ReLU(),
            nn.Conv1D(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            nn.BatchNorm1D(mid_channels),
            nn.ReLU(),
        )

        self.out_conv = nn.Sequential(
            nn.Conv1D(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=True,
            ),
            nn.BatchNorm1D(out_channels),
            nn.ReLU(),
        )
        self.merger.apply(param_init.init_weight)
        self.reduce_conv.apply(param_init.init_weight)
        self.conv1.apply(param_init.init_weight)
        self.conv2.apply(param_init.init_weight)
        self.out_conv.apply(param_init.init_weight)

    def forward(self, x, pe=None):
        # [N,C,H,W]
        if pe is not None:
            x = self.merger(paddle.concat([x, pe], 1))
        else:
            x = self.merger(x)
        x = x.max(2)
        x = self.reduce_conv(x)
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        x = self.out_conv(x)
        return x


class DepthReducer(nn.Layer):
    def __init__(self, img_channels, mid_channels):
        """Module that compresses the predicted
            categorical depth in height dimension

        Args:
            img_channels (int): in_channels
            mid_channels (int): mid_channels
        """
        super().__init__()
        self.vertical_weighter = nn.Sequential(
            nn.Conv2D(
                img_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
            nn.Conv2D(mid_channels, 1, kernel_size=3, stride=1, padding=1),
        )
        self.vertical_weighter.apply(param_init.init_weight)

    def forward(self, feat, depth):
        vert_weight = F.softmax(self.vertical_weighter(feat), 2)  # [N,1,H,W]
        depth = (depth * vert_weight).sum(2)
        return depth
