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

import paddle
from paddle import nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn import functional as F
import paddle.distributed as dist

from paddle3d.models.losses import IOULoss, sigmoid_focal_loss
from paddle3d.models.layers import LayerListDial, Scale, FrozenBatchNorm2d, param_init
from paddle3d.apis import manager

__all__ = ["FCOS2DHead", "FCOS2DLoss", "FCOS2DInference"]


@manager.HEADS.add_component
class FCOS2DHead(nn.Layer):
    """
    This code is based on https://github.com/TRI-ML/dd3d/blob/main/tridet/modeling/dd3d/fcos2d.py#L30
    """

    def __init__(self,
                 in_strides,
                 in_channels,
                 num_classes=5,
                 use_scale=True,
                 box2d_scale_init_factor=1.0,
                 version="v2",
                 num_cls_convs=4,
                 num_box_convs=4,
                 use_deformable=False,
                 norm="BN"):
        super().__init__()
        self.in_strides = in_strides
        self.num_levels = len(in_strides)
        self.num_classes = num_classes
        self.use_scale = use_scale
        self.box2d_scale_init_factor = box2d_scale_init_factor
        self.version = version

        assert len(
            set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        if use_deformable:
            raise ValueError("Not supported yet.")

        head_configs = {'cls': num_cls_convs, 'box2d': num_box_convs}

        for head_name, num_convs in head_configs.items():
            tower = []
            if self.version == "v1":
                for _ in range(num_convs):
                    conv_func = nn.Conv2D
                    tower.append(
                        conv_func(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True))
                    if norm == "BN":
                        tower.append(
                            LayerListDial([
                                nn.BatchNorm2D(
                                    in_channels,
                                    weight_attr=ParamAttr(
                                        regularizer=L2Decay(0.0)))
                                for _ in range(self.num_levels)
                            ]))
                    else:
                        raise NotImplementedError()
                    tower.append(nn.ReLU())
            elif self.version == "v2":
                for _ in range(num_convs):
                    # Each FPN level has its own batchnorm layer.
                    # "BN" is converted to "SyncBN" in distributed training
                    if norm == "BN":
                        norm_layer = LayerListDial([
                            nn.BatchNorm2D(
                                in_channels,
                                weight_attr=ParamAttr(regularizer=L2Decay(0.0)))
                            for _ in range(self.num_levels)
                        ])
                    elif norm == "FrozenBN":
                        norm_layer = LayerListDial([
                            FrozenBatchNorm2d(in_channels)
                            for _ in range(self.num_levels)
                        ])
                    else:
                        raise NotImplementedError()
                    tower.append(
                        nn.Conv2D(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias_attr=False))
                    tower.append(norm_layer)
                    tower.append(nn.ReLU())
            else:
                raise ValueError(f"Invalid FCOS2D version: {self.version}")
            self.add_sublayer(f'{head_name}_tower', nn.Sequential(*tower))

        self.cls_logits = nn.Conv2D(
            in_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.box2d_reg = nn.Conv2D(
            in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2D(
            in_channels, 1, kernel_size=3, stride=1, padding=1)

        if self.use_scale:
            if self.version == "v1":
                self.scales_reg = nn.LayerList([
                    Scale(init_value=stride * self.box2d_scale_init_factor)
                    for stride in self.in_strides
                ])
            else:
                self.scales_box2d_reg = nn.LayerList([
                    Scale(init_value=stride * self.box2d_scale_init_factor)
                    for stride in self.in_strides
                ])

        self.init_weights()

    def init_weights(self):

        for tower in [self.cls_tower, self.box2d_tower]:
            for l in tower.sublayers():
                if isinstance(l, nn.Conv2D):
                    param_init.kaiming_normal_init(
                        l.weight, mode='fan_out', nonlinearity='relu')
                    if l.bias is not None:
                        param_init.constant_init(l.bias, value=0.0)

        predictors = [self.cls_logits, self.box2d_reg, self.centerness]

        for layers in predictors:
            for l in layers.sublayers():
                if isinstance(l, nn.Conv2D):
                    param_init.kaiming_uniform_init(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        param_init.constant_init(l.bias, value=0.0)

    def forward(self, x):
        logits = []
        box2d_reg = []
        centerness = []

        extra_output = {"cls_tower_out": []}

        for l, feature in enumerate(x):
            cls_tower_out = self.cls_tower(feature)
            bbox_tower_out = self.box2d_tower(feature)

            # 2D box
            logits.append(self.cls_logits(cls_tower_out))
            centerness.append(self.centerness(bbox_tower_out))
            box_reg = self.box2d_reg(bbox_tower_out)
            if self.use_scale:
                # TODO: to optimize the runtime, apply this scaling in inference (and loss compute) only on FG pixels?
                if self.version == "v1":
                    box_reg = self.scales_reg[l](box_reg)
                else:
                    box_reg = self.scales_box2d_reg[l](box_reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            box2d_reg.append(F.relu(box_reg))

            extra_output['cls_tower_out'].append(cls_tower_out)

        return logits, box2d_reg, centerness, extra_output


def reduce_sum(tensor):
    if not dist.get_world_size() > 1:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor)
    return tensor


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, 0::2]
    top_bottom = reg_targets[:, 1::2]
    ctrness = (left_right.min(axis=-1) / left_right.max(axis=-1)) * \
                 (top_bottom.min(axis=-1) / top_bottom.max(axis=-1))
    return paddle.sqrt(ctrness)


@manager.LOSSES.add_component
class FCOS2DLoss(nn.Layer):
    """
    This code is based on https://github.com/TRI-ML/dd3d/blob/main/tridet/modeling/dd3d/fcos2d.py#L159
    """

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 loc_loss_type='giou',
                 num_classes=5):
        super().__init__()
        self.focal_loss_alpha = alpha
        self.focal_loss_gamma = gamma

        self.box2d_reg_loss_fn = IOULoss(loc_loss_type)

        self.num_classes = num_classes

    def forward(self, logits, box2d_reg, centerness, targets):
        labels = targets['labels']
        box2d_reg_targets = targets['box2d_reg_targets']
        pos_inds = targets["pos_inds"]

        if len(labels) != box2d_reg_targets.shape[0]:
            raise ValueError(
                f"The size of 'labels' and 'box2d_reg_targets' does not match: a={len(labels)}, b={box2d_reg_targets.shape[0]}"
            )

        # Flatten predictions
        logits = paddle.concat([
            x.transpose([0, 2, 3, 1]).reshape([-1, self.num_classes])
            for x in logits
        ],
                               axis=0)
        box2d_reg_pred = paddle.concat(
            [x.transpose([0, 2, 3, 1]).reshape([-1, 4]) for x in box2d_reg],
            axis=0)
        centerness_pred = paddle.concat(
            [x.transpose([0, 2, 3, 1]).reshape([-1]) for x in centerness],
            axis=0)

        # Classification loss
        num_pos_local = pos_inds.numel()
        num_gpus = dist.get_world_size()
        total_num_pos = reduce_sum(paddle.to_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        cls_target = paddle.zeros_like(logits)
        if num_pos_local > 0:
            cls_target[pos_inds, labels[pos_inds]] = 1

        loss_cls = sigmoid_focal_loss(
            logits,
            cls_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg

        if pos_inds.numel() == 0:
            losses = {
                "loss_cls": loss_cls,
                "loss_box2d_reg": box2d_reg_pred.sum() * 0.,
                "loss_centerness": centerness_pred.sum() * 0.,
            }
            return losses, {}

        # NOTE: The rest of losses only consider foreground pixels.
        if num_pos_local == 1:
            box2d_reg_pred = box2d_reg_pred[pos_inds].unsqueeze(0)
            box2d_reg_targets = box2d_reg_targets[pos_inds].unsqueeze(0)
        else:
            box2d_reg_pred = box2d_reg_pred[pos_inds]
            box2d_reg_targets = box2d_reg_targets[pos_inds]
        centerness_pred = centerness_pred[pos_inds]

        # Compute centerness targets here using 2D regression targets of foreground pixels.
        centerness_targets = compute_ctrness_targets(box2d_reg_targets)

        # Denominator for all foreground losses.
        ctrness_targets_sum = centerness_targets.sum()
        loss_denom = max(
            reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)

        # 2D box regression loss
        loss_box2d_reg = self.box2d_reg_loss_fn(
            box2d_reg_pred, box2d_reg_targets, centerness_targets) / loss_denom

        # Centerness loss
        loss_centerness = F.binary_cross_entropy_with_logits(
            centerness_pred, centerness_targets, reduction="sum") / num_pos_avg

        loss_dict = {
            "loss_cls": loss_cls,
            "loss_box2d_reg": loss_box2d_reg,
            "loss_centerness": loss_centerness
        }
        extra_info = {
            "loss_denom": loss_denom,
            "centerness_targets": centerness_targets
        }

        return loss_dict, extra_info


@manager.MODELS.add_component
class FCOS2DInference():
    """
    This code is based on https://github.com/TRI-ML/dd3d/blob/main/tridet/modeling/dd3d/fcos2d.py#L242
    """

    def __init__(self,
                 thresh_with_ctr=True,
                 pre_nms_thresh=0.05,
                 pre_nms_topk=1000,
                 post_nms_topk=100,
                 nms_thresh=0.75,
                 num_classes=5):
        self.thresh_with_ctr = thresh_with_ctr
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.num_classes = num_classes

    def __call__(self, logits, box2d_reg, centerness, locations):

        pred_instances = []  # List[List[dict]], shape = (L, B)
        extra_info = []
        for lvl, (logits_lvl, box2d_reg_lvl, centerness_lvl, locations_lvl) in \
            enumerate(zip(logits, box2d_reg, centerness, locations)):

            instances_per_lvl, extra_info_per_lvl = self.forward_for_single_feature_map(
                logits_lvl, box2d_reg_lvl, centerness_lvl,
                locations_lvl)  # List of dict; one for each image.

            for instances_per_im in instances_per_lvl:
                instances_per_im['fpn_levels'] = paddle.ones(
                    [instances_per_im['pred_boxes'].shape[0]], dtype='float64')
                if instances_per_im['pred_boxes'].shape[0] != 0:
                    instances_per_im['fpn_levels'] *= lvl

            pred_instances.append(instances_per_lvl)
            extra_info.append(extra_info_per_lvl)

        return pred_instances, extra_info

    def forward_for_single_feature_map(self, logits, box2d_reg, centerness,
                                       locations):
        N, C, _, __ = logits.shape

        # put in the same format as locations
        scores = F.sigmoid(logits.transpose([0, 2, 3, 1]).reshape([N, -1, C]))
        box2d_reg = box2d_reg.transpose([0, 2, 3, 1]).reshape([N, -1, 4])
        centerness = F.sigmoid(
            centerness.transpose([0, 2, 3, 1]).reshape([N, -1]))

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            scores = scores * centerness[:, :, None]

        candidate_mask = scores > self.pre_nms_thresh

        pre_nms_topk = candidate_mask.reshape([N, -1]).sum(1)
        pre_nms_topk = pre_nms_topk.clip(max=self.pre_nms_topk)

        if not self.thresh_with_ctr:
            scores = scores * centerness[:, :, None]

        results = []
        all_fg_inds_per_im, all_topk_indices, all_class_inds_per_im = [], [], []
        for i in range(N):
            scores_per_im = scores[i]
            candidate_mask_per_im = candidate_mask[i]
            scores_per_im = scores_per_im[candidate_mask_per_im]

            candidate_inds_per_im = candidate_mask_per_im.nonzero(
                as_tuple=False)
            fg_inds_per_im = candidate_inds_per_im[:, 0]
            class_inds_per_im = candidate_inds_per_im[:, 1]

            # Cache info here.
            all_fg_inds_per_im.append(fg_inds_per_im)
            all_class_inds_per_im.append(class_inds_per_im)

            if fg_inds_per_im.shape[0] == 0:
                box2d_reg_per_im = paddle.zeros([0, 4])
                locations_per_im = paddle.zeros([0, 2])
            else:
                box2d_reg_per_im = box2d_reg[i][fg_inds_per_im]
                locations_per_im = locations[fg_inds_per_im]

            pre_nms_topk_per_im = pre_nms_topk[i]

            if candidate_mask_per_im.sum().item() > pre_nms_topk_per_im.item():
                scores_per_im, topk_indices = \
                    scores_per_im.topk(pre_nms_topk_per_im, sorted=False)

                class_inds_per_im = class_inds_per_im[topk_indices]
                box2d_reg_per_im = box2d_reg_per_im[topk_indices]
                locations_per_im = locations_per_im[topk_indices]
            else:
                topk_indices = None

            all_topk_indices.append(topk_indices)

            if locations_per_im.shape[0] == 0:
                detections = paddle.zeros([0, 4])
            elif len(locations_per_im.shape) == 1:
                locations_per_im = locations_per_im.unsqueeze(0)
                box2d_reg_per_im = box2d_reg_per_im.unsqueeze(0)
                detections = paddle.stack([
                    locations_per_im[:, 0] - box2d_reg_per_im[:, 0],
                    locations_per_im[:, 1] - box2d_reg_per_im[:, 1],
                    locations_per_im[:, 0] + box2d_reg_per_im[:, 2],
                    locations_per_im[:, 1] + box2d_reg_per_im[:, 3],
                ],
                                          axis=1)
            else:
                detections = paddle.stack([
                    locations_per_im[:, 0] - box2d_reg_per_im[:, 0],
                    locations_per_im[:, 1] - box2d_reg_per_im[:, 1],
                    locations_per_im[:, 0] + box2d_reg_per_im[:, 2],
                    locations_per_im[:, 1] + box2d_reg_per_im[:, 3],
                ],
                                          axis=1)

            instances = {}
            instances['pred_boxes'] = detections
            if scores_per_im.shape[0] == 0:
                instances['scores'] = scores_per_im
            else:
                instances['scores'] = paddle.sqrt(scores_per_im)
            instances['pred_classes'] = class_inds_per_im
            instances['locations'] = locations_per_im

            results.append(instances)

        extra_info = {
            "fg_inds_per_im": all_fg_inds_per_im,
            "class_inds_per_im": all_class_inds_per_im,
            "topk_indices": all_topk_indices
        }
        return results, extra_info

    def nms_and_top_k(self, instances_per_im, score_key_for_nms="scores"):
        results = []
        for instances in instances_per_im:
            if self.nms_thresh > 0:
                # Multiclass NMS.
                if instances['pred_boxes'].shape[0] == 0:
                    results.append(instances)
                    continue
                keep = paddle.vision.ops.nms(
                    boxes=instances['pred_boxes'],
                    iou_threshold=self.nms_thresh,
                    scores=instances[score_key_for_nms],
                    category_idxs=instances['pred_classes'],
                    categories=[0, 1, 2, 3, 4])
                if keep.shape[0] == 0:
                    instances['pred_boxes'] = paddle.zeros([0, 4])
                    instances['pred_classes'] = paddle.zeros([0])
                    instances['scores'] = paddle.zeros([0])
                    instances['scores_3d'] = paddle.zeros([0])
                    instances['pred_boxes3d'] = paddle.zeros([0, 10])
                instances['pred_boxes'] = instances['pred_boxes'][keep]
                instances['pred_classes'] = instances['pred_classes'][keep]
                instances['scores'] = instances['scores'][keep]
                instances['scores_3d'] = instances['scores_3d'][keep]
                instances['pred_boxes3d'] = instances['pred_boxes3d'][keep]
                if len(instances['pred_boxes'].shape) == 1:
                    instances['pred_boxes'] = instances['pred_boxes'].unsqueeze(
                        0)
                    instances['pred_boxes3d'] = instances[
                        'pred_boxes3d'].unsqueeze(0)
            num_detections = instances['pred_boxes3d'].shape[0]

            # Limit to max_per_image detections **over all classes**
            if num_detections > self.post_nms_topk > 0:
                scores = instances['scores']
                image_thresh, _ = paddle.kthvalue(
                    scores, num_detections - self.post_nms_topk + 1)
                keep = scores >= image_thresh.item()
                keep = paddle.nonzero(keep).squeeze(1)
                instances['pred_boxes'] = instances['pred_boxes'][keep]
                instances['pred_classes'] = instances['pred_classes'][keep]
                instances['scores'] = instances['scores'][keep]
                instances['scores_3d'] = instances['scores_3d'][keep]
                instances['pred_boxes3d'] = instances['pred_boxes3d'][keep]
                if len(instances['pred_boxes'].shape) == 1:
                    instances['pred_boxes'].unsqueeze(0)
                    instances['pred_boxes3d'].unsqueeze(0)
            results.append(instances)
        return results
