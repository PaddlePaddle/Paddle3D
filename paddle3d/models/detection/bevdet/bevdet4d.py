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

from typing import Dict, List
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle3d.apis import manager


class DictObject(Dict):
    def __init__(self, config: Dict):
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, DictObject(value))
            else:
                setattr(self, key, value)


@manager.MODELS.add_component
class BEVDet4D(nn.Layer):
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
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(BEVDet4D, self).__init__(**kwargs)

        self.pts_bbox_head = pts_bbox_head
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.with_img_neck = self.img_neck is not None
        self.img_view_transformer = img_view_transformer
        self.img_bev_encoder_backbone = img_bev_encoder_backbone
        self.img_bev_encoder_neck = img_bev_encoder_neck

        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = pre_process

        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1
        self.with_prev = with_prev
        self.start_temporal_epoch = start_temporal_epoch

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
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

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
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.reshape((B, N, output_dim, ouput_H, output_W))
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
            trans_prev = paddle.concat(rots[1:], axis=0)
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
        """
        points = None
        img_metas = samples['meta']
        img = samples['img_inputs']
        gt_depth = samples['gt_depth']
        gt_labels_3d = samples['gt_labels_3d']
        gt_bboxes_3d = samples['gt_bboxes_3d']
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)
        return {"loss": losses}

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats[0])  # single apply
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs[0]]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward(self, samples, *args, **kwargs):

        if self.training:
            return self.forward_train(samples, *args, **kwargs)

        self.align_after_view_transfromation = True
        return self.forward_test(samples, *args, **kwargs)

    def forward_test(self, samples, **kwargs):
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

    def simple_test(self, samples, img=None, rescale=False, **kwargs):

        points = None
        img_metas = samples['meta']
        img = samples['img_inputs']
        """Test function without augmentation."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return {"preds": bbox_list}

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x[0])  # single apply
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs[0], img_metas, rescale=rescale)
        bbox_results = [
            self.bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def bbox3d2result(self, bboxes, scores, labels, attrs=None):
        """Convert detection results to a list of numpy arrays.
        """
        result_dict = dict(
            boxes_3d=bboxes.numpy(),
            scores_3d=scores.numpy(),
            labels_3d=labels.numpy())

        if attrs is not None:
            result_dict['attrs_3d'] = attrs

        return result_dict

    def forward_dummy(self, samples, **kwargs):
        points = None
        img_metas = [samples['meta']]
        img_inputs = samples['img_inputs']
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs[0], img_metas=img_metas, **kwargs)
        outs = self.pts_bbox_head(img_feats[0])
        return outs
