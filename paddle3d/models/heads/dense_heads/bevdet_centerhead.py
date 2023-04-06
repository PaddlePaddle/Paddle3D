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
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddle3d.apis import manager
from paddle3d.geometries.bbox import circle_nms
from paddle3d.models.backbones.second_backbone import build_conv_layer
from paddle3d.models.layers.layer_libs import rotate_nms_pcdet
from paddle3d.models.losses import GaussianFocalLoss, L1Loss
from paddle3d.models.voxel_encoders.pillar_encoder import build_norm_layer
from paddle3d.utils.logger import logger
import paddle.distributed as dist


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
            bias=False,
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
        """
        ret_dict = dict()
        for head in self.heads.keys():
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@manager.MODELS.add_component
class CenterHeadMatch(nn.Layer):
    def __init__(self,
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
                 bbox_coder=None,
                 norm_bbox=True,
                 test_cfg=None,
                 train_cfg=None,
                 task_specific=True):
        super(CenterHeadMatch, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight  # weight between hm loss and loc loss

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.loss_cls = GaussianFocalLoss(reduction='mean')
        self.loss_bbox = L1Loss(reduction='mean', loss_weight=0.25)

        self.box_n_dim = 9 if 'vel' in common_heads else 7
        self.with_velocity = True if 'vel' in common_heads else False
        self.code_weights = code_weights
        self.use_direction_classifier = False
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.bbox_coder = bbox_coder
        self.norm_bbox = norm_bbox
        self.task_specific = task_specific

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg)

        self.task_heads = nn.LayerList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_hm_conv)))
            self.task_heads.append(
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

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts, x

    def _sigmoid(self, x):
        y = paddle.clip(F.sigmoid(x), min=1e-4, max=1 - 1e-4)
        return y

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.
        """
        heatmaps = []
        anno_boxes = []
        inds = []
        masks = []
        for gt_bbox_3d, gt_label_3d in zip(gt_bboxes_3d, gt_labels_3d):
            heatmap, annos_box, ind, mask = self.get_targets_single(
                gt_bbox_3d, gt_label_3d)
            heatmaps.append(heatmap)
            anno_boxes.append(annos_box)
            inds.append(ind)
            masks.append(mask)

        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [paddle.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [paddle.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [paddle.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [paddle.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """

        def gravity_center(box):
            bottom_center = box[:, :3]
            gravity_center = paddle.zeros(bottom_center.shape,
                                          bottom_center.dtype)
            gravity_center[:, :2] = bottom_center[:, :2]
            gravity_center[:, 2] = bottom_center[:, 2] + box[:, 5] * 0.5
            return gravity_center

        gt_gravity_centers = gravity_center(gt_bboxes_3d)

        gt_bboxes_3d = paddle.concat((gt_gravity_centers, gt_bboxes_3d[:, 3:]),
                                     axis=1)

        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = paddle.to_tensor(self.train_cfg['grid_size'])
        pc_range = paddle.to_tensor(self.train_cfg['point_cloud_range'])
        voxel_size = paddle.to_tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                paddle.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                if m[0].shape[0] != 0:
                    task_box.append(gt_bboxes_3d[m])
                    # 0 is background for each task, so we need to add 1 here.
                    task_class.append(gt_labels_3d[m] + 1 - flag2)

            task_boxes.append(
                paddle.concat(task_box, axis=0).
                squeeze(1) if task_box != [] else paddle.empty((0, 1, 9)))
            task_classes.append(
                paddle.concat(task_class).cast("int64").
                squeeze(1) if task_box != [] else paddle.empty((0, 1)))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = paddle.zeros((len(self.class_names[idx]),
                                    feature_map_size[1], feature_map_size[0]),
                                   dtype=gt_bboxes_3d.dtype)

            if self.with_velocity:
                anno_box = paddle.zeros((max_objs, 10), dtype='float32')
            else:
                anno_box = paddle.zeros((max_objs, 8), dtype='float32')

            ind = paddle.zeros((max_objs, ), dtype='int64')
            mask = paddle.zeros((max_objs, ), dtype='int32')

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = (task_classes[idx][k] - 1).item()

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg['out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = paddle.to_tensor([coor_x, coor_y], dtype='float32')
                    center_int = center.cast("int32")

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    heatmap[cls_id] = draw_gaussian(heatmap[cls_id], center_int,
                                                    radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = (y * feature_map_size[0] + x).cast("int64")
                    mask[new_idx] = 1
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = paddle.log(box_dim)
                    if self.with_velocity:
                        vx, vy = task_boxes[idx][k][7:]
                        anno_box[new_idx] = paddle.concat(
                            [(center - paddle.to_tensor([x, y])).squeeze(), z,
                             box_dim.cast('float32'),
                             paddle.sin(rot)[0],
                             paddle.cos(rot)[0], vx, vy])
                    else:
                        anno_box[new_idx] = paddle.concat(
                            [(center - paddle.to_tensor([x, y])).squeeze(), z,
                             box_dim.cast('float32'),
                             paddle.sin(rot),
                             paddle.cos(rot)])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        if not self.task_specific:
            loss_dict['loss'] = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['heatmap'] = self._sigmoid(preds_dict['heatmap'])
            num_pos = (heatmaps[task_id] == 1).cast("float32").sum()

            cls_avg_factor = paddle.clip(
                reduce_mean(paddle.to_tensor(num_pos, heatmaps[task_id].dtype)),
                min=1).item()

            loss_heatmap = self.loss_cls(
                preds_dict['heatmap'],
                heatmaps[task_id],
                avg_factor=cls_avg_factor)

            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict['anno_box'] = paddle.concat(
                (
                    preds_dict['reg'],
                    preds_dict['height'],
                    preds_dict['dim'],
                    preds_dict['rot'],
                    preds_dict['vel'],
                ),
                axis=1,
            )
            # Regression loss for dimension, offset, height, rotation
            num = masks[task_id].cast("float32").sum()  #.float().sum()
            ind = inds[task_id]
            pred = preds_dict['anno_box'].transpose((0, 2, 3, 1))
            pred = pred.reshape((pred.shape[0], -1, pred.shape[3]))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).cast(
                "float32")
            num = paddle.clip(
                reduce_mean(paddle.to_tensor(num, dtype=target_box.dtype)),
                min=1e-4).item()
            isnotnan = (~paddle.isnan(target_box)).cast('float32')
            mask *= isnotnan
            code_weights = self.train_cfg['code_weights']

            bbox_weights = mask * paddle.to_tensor(
                code_weights, dtype=mask.dtype)
            if self.task_specific:
                name_list = ['xy', 'z', 'whl', 'yaw', 'vel']
                clip_index = [0, 2, 3, 6, 8, 10]
                for reg_task_id in range(len(name_list)):
                    pred_tmp = pred[..., clip_index[reg_task_id]:clip_index[
                        reg_task_id + 1]]
                    target_box_tmp = target_box[..., clip_index[reg_task_id]:
                                                clip_index[reg_task_id + 1]]
                    bbox_weights_tmp = bbox_weights[
                        ..., clip_index[reg_task_id]:clip_index[reg_task_id +
                                                                1]]
                    loss_bbox_tmp = self.loss_bbox(
                        pred_tmp,
                        target_box_tmp,
                        bbox_weights_tmp,
                        avg_factor=(num + 1e-4))
                    loss_dict[f'task{task_id}.loss_%s' %
                              (name_list[reg_task_id])] = loss_bbox_tmp
                loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            else:
                loss_bbox = self.loss_bbox(
                    pred, target_box, bbox_weights, avg_factor=num)
                loss_dict['loss'] += loss_bbox
                loss_dict['loss'] += loss_heatmap

        return loss_dict

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.
        """
        dim = feat.shape[2]
        ind = ind.unsqueeze(2).expand((ind.shape[0], ind.shape[1], dim))
        feat = feat.take_along_axis(ind, 1)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.reshape(-1, dim)
        return feat

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

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict['heatmap'].shape[0]
            batch_heatmap = F.sigmoid(preds_dict['heatmap'])

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if self.norm_bbox:
                batch_dim = paddle.exp(preds_dict['dim'])
            else:
                batch_dim = preds_dict['dim']

            batch_rots = preds_dict['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            nms_type = self.test_cfg.get('nms_type')
            if isinstance(nms_type, list):
                nms_type = nms_type[task_id]
            if nms_type == 'circle':
                ret_task = []
                for i in range(batch_size):

                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    assert boxes3d.shape[0] == scores.shape[0]
                    if boxes3d.shape[0] == 0:
                        ret = dict(
                            bboxes=paddle.empty_like(boxes3d),
                            scores=paddle.empty_like(scores),
                            labels=paddle.empty_like(labels))
                        ret_task.append(ret)
                        continue
                    centers = boxes3d[:, :2]
                    boxes = paddle.concat(
                        [centers, scores.reshape((-1, 1))], axis=1)
                    keep = paddle.to_tensor(
                        _circle_nms(
                            boxes,
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype='int64')

                    boxes3d = boxes3d.index_select(keep)
                    scores = scores.index_select(keep)
                    labels = labels.index_select(keep)
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg, batch_cls_preds,
                                             batch_reg_preds, batch_cls_labels,
                                             img_metas, task_id))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []

        for i in range(num_samples):
            bboxes = paddle.empty((0, 9))
            scores = paddle.empty((0, ))
            labels = paddle.empty((0, ))
            for k in rets[0][i].keys():

                if k == 'bboxes':
                    for ret in rets:
                        if ret[i][k].shape[0] != 0:
                            bboxes = paddle.concat([bboxes, ret[i][k]])
                    if bboxes.shape[0] != 0:
                        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                elif k == 'scores':
                    for ret in rets:
                        if ret[i][k].shape[0] != 0:
                            scores = paddle.concat([scores, ret[i][k]])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    for ret in rets:
                        if ret[i][k].shape[0] != 0:
                            labels = paddle.concat(
                                [labels, ret[i][k].cast("int32")])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas,
                            task_id):
        """Rotate nms for each task.
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = paddle.to_tensor(
                post_center_range, dtype=batch_reg_preds[0].dtype)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):
            if box_preds.shape[0] != 0:
                default_val = [1.0 for _ in range(len(self.task_heads))]
                factor = self.test_cfg.get('nms_rescale_factor',
                                           default_val)[task_id]
                if isinstance(factor, list):
                    for cid in range(len(factor)):
                        box_preds_np_tmp = box_preds.numpy()
                        box_preds_np_tmp[(
                            cls_labels == cid).numpy(), 3:6] = box_preds_np_tmp[
                                (cls_labels == cid).numpy(), 3:6] * factor[cid]
                        box_preds = paddle.to_tensor(box_preds_np_tmp)

                else:
                    box_preds[:, 3:6] = box_preds[:, 3:6] * factor

                # Apply NMS in birdeye view

                # get the highest score per prediction, then apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = cls_preds
                    top_labels = paddle.zeros((cls_preds.shape[0], ),
                                              dtype='int64')

                else:
                    top_labels = cls_labels.cast("int64")
                    top_scores = cls_preds

                if self.test_cfg['score_threshold'] > 0.0:
                    thresh = paddle.to_tensor([
                        self.test_cfg['score_threshold']
                    ]).cast(cls_preds.dtype)

                    top_scores_keep = top_scores >= thresh
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if self.test_cfg['score_threshold'] > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]

                    boxes_for_nms = box_preds.clone()

                    # the nms in 3d detection just remove overlap boxes.
                    if isinstance(self.test_cfg['nms_thr'], list):
                        nms_thresh = self.test_cfg['nms_thr'][task_id]
                    else:
                        nms_thresh = self.test_cfg['nms_thr']
                    selected = nms_bev(
                        boxes_for_nms,
                        top_scores,
                        thresh=nms_thresh,
                        pre_max_size=self.test_cfg['pre_max_size'],
                        post_max_size=self.test_cfg['post_max_size'])
                else:
                    selected = []

                if isinstance(factor, list):
                    for cid in range(len(factor)):
                        box_preds_np_tmp = box_preds.numpy()
                        box_preds_np_tmp[(
                            cls_labels == cid).numpy(), 3:6] = box_preds_np_tmp[
                                (cls_labels == cid).numpy(), 3:6] / factor[cid]
                        box_preds = paddle.to_tensor(box_preds_np_tmp)
                else:
                    box_preds[:, 3:6] = box_preds[:, 3:6] / factor

                selected_boxes = box_preds.index_select(selected)
                selected_labels = top_labels.index_select(selected)
                selected_scores = top_scores.index_select(selected)

                # finally generate predictions.
                if selected_boxes.shape[0] != 0:
                    box_preds = selected_boxes
                    scores = selected_scores
                    label_preds = selected_labels
                    final_box_preds = box_preds
                    final_scores = scores
                    final_labels = label_preds
                    if post_center_range is not None:
                        mask = (final_box_preds[:, :3] >=
                                post_center_range[:3]).all(1)
                        mask &= (final_box_preds[:, :3] <=
                                 post_center_range[3:]).all(1)
                        predictions_dict = dict(
                            bboxes=final_box_preds[mask],
                            scores=final_scores[mask],
                            labels=final_labels[mask])
                    else:
                        predictions_dict = dict(
                            bboxes=final_box_preds,
                            scores=final_scores,
                            labels=final_labels)
                else:
                    dtype = batch_reg_preds[0].dtype
                    predictions_dict = dict(
                        bboxes=paddle.zeros([0, self.bbox_coder.code_size],
                                            dtype=dtype),
                        scores=paddle.zeros([0], dtype=dtype),
                        labels=paddle.zeros([0], dtype='int64'))
            else:
                dtype = batch_reg_preds[0].dtype
                predictions_dict = dict(
                    bboxes=paddle.zeros([0, self.bbox_coder.code_size],
                                        dtype=dtype),
                    scores=paddle.zeros([0], dtype=dtype),
                    labels=paddle.zeros([0], dtype='int64'))
            predictions_dicts.append(predictions_dict)
        return predictions_dicts


import numpy as np


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.numpy(),
                               thresh=min_radius))[:post_max_size]

    keep = paddle.to_tensor(keep, dtype='int32')

    return keep


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.
    """
    boxes = paddle.zeros(boxes_xywhr.shape)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]
    return boxes


def nms_bev(boxes, scores, thresh, pre_max_size=None, post_max_size=None):
    """NMS function GPU implementation (for BEV boxes). The overlap of two
    boxes for IoU calculation is defined as the exact overlapping area of the
    two boxes. In this function, one can also set ``pre_max_size`` and
    ``post_max_size``.

    """

    order = scores.argsort(descending=True)
    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes.index_select(order)
    scores = scores.index_select(order)

    boxes_copy = boxes.clone()

    boxes[:, 6] = -boxes[:, 6] - np.pi / 2
    boxes[:, 4] = boxes_copy[:, 3]
    boxes[:, 3] = boxes_copy[:, 4]

    keep = rotate_nms_pcdet(
        boxes[:, :7],
        scores,
        thresh,
        pre_max_size=pre_max_size,
        post_max_size=post_max_size)
    keep = order[keep]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = paddle.to_tensor(
        gaussian[radius - top:radius + bottom, radius - left:radius + right],
        dtype='float32')

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        heatmap[y - top:y + bottom, x - left:x + right] = paddle.maximum(
            masked_heatmap, masked_gaussian * k)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float, optional): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = paddle.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = paddle.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = paddle.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(
        tensor.scale_(1. / dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


@manager.MODELS.add_component
class CenterPointBBoxCoder(object):
    """Bbox coder for CenterPoint.

    """

    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 code_size=9):

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size

    def _gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.
        """
        dim = feats.shape[2]
        inds = inds.unsqueeze(2).expand((inds.shape[0], inds.shape[1], dim))
        feats = feats.take_along_axis(inds, 1)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.reshape((-1, dim))
        return feats

    def _topk(self, scores, K=80):
        """Get indexes based on scores.
        """
        batch, cat, height, width = scores.shape

        topk_scores, topk_inds = paddle.topk(
            scores.reshape((batch, cat, -1)), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds.cast("float32") / paddle.to_tensor(
            width, dtype='float32')).cast('int32').cast('float32')
        topk_xs = (topk_inds % width).cast("int32").cast("float32")

        topk_score, topk_ind = paddle.topk(topk_scores.reshape((batch, -1)), K)
        topk_clses = (
            topk_ind / paddle.to_tensor(K, dtype='float32')).cast('int32')
        topk_inds = self._gather_feat(
            topk_inds.reshape((batch, -1, 1)), topk_ind).reshape((batch, K))
        topk_ys = self._gather_feat(topk_ys.reshape((batch, -1, 1)),
                                    topk_ind).reshape((batch, K))
        topk_xs = self._gather_feat(topk_xs.reshape((batch, -1, 1)),
                                    topk_ind).reshape((batch, K))

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.
        """
        feat = feat.transpose((0, 2, 3, 1))
        feat = feat.reshape((feat.shape[0], -1, feat.shape[3]))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        pass

    def decode(self,
               heat,
               rot_sine,
               rot_cosine,
               hei,
               dim,
               vel,
               reg=None,
               task_id=-1):
        """Decode bboxes.
        """
        batch, cat, _, _ = heat.shape

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.reshape((batch, self.max_num, 2))
            xs = xs.reshape((batch, self.max_num, 1)) + reg[:, :, 0:1]
            ys = ys.reshape((batch, self.max_num, 1)) + reg[:, :, 1:2]
        else:
            xs = xs.reshape((batch, self.max_num, 1)) + 0.5
            ys = ys.reshape((batch, self.max_num, 1)) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.reshape((batch, self.max_num, 1))

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.reshape((batch, self.max_num, 1))
        rot = paddle.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.reshape((batch, self.max_num, 1))

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.reshape((batch, self.max_num, 3))

        # class label
        clses = clses.reshape((batch, self.max_num)).reshape(
            (batch, self.max_num)).cast("float32")
        scores = scores.reshape((batch, self.max_num))

        xs = xs.reshape(
            (batch, self.max_num,
             1)) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys.reshape(
            (batch, self.max_num,
             1)) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = paddle.concat([xs, ys, hei, dim, rot], axis=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.reshape((batch, self.max_num, 2))
            final_box_preds = paddle.concat([xs, ys, hei, dim, rot, vel],
                                            axis=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = paddle.to_tensor(self.post_center_range)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i][cmask]
                scores = final_scores[i][cmask]
                labels = final_preds[i][cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts
