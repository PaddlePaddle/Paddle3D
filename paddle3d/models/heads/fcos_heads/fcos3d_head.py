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
import paddle.nn.functional as F

from paddle3d.models.losses import DisentangledBox3DLoss, unproject_points2d
from paddle3d.models.layers import LayerListDial, Offset, Scale, FrozenBatchNorm2d, param_init
from paddle3d.apis import manager
from paddle3d.utils.logger import logger
from paddle3d.utils.transform import matrix_to_quaternion, quaternion_to_matrix

__all__ = ["FCOS3DHead", "FCOS3DLoss", "FCOS3DInference"]

PI = 3.14159265358979323846
EPS = 1e-7


def allocentric_to_egocentric(quat, proj_ctr, inv_intrinsics):
    """
    Args:
        quat (paddle.Tensor with shape (N, 4)): Batch of (allocentric) quaternions.
        proj_ctr (paddle.Tensor with shape (N, 2)): Projected centers. xy coordninates.
        inv_intrinsics (paddle.Tensor with shape (N, 3, 3)): Inverted intrinsics.
    """
    R_obj_to_local = quaternion_to_matrix(quat)

    # ray == z-axis in local orientaion
    ray = unproject_points2d(proj_ctr, inv_intrinsics)
    z = ray / paddle.linalg.norm(ray, axis=1, keepdim=True)

    # gram-schmit process: local_y = global_y - global_y \dot local_z
    y = paddle.to_tensor([[0., 1., 0.]]) - z[:, 1:2] * z
    y = y / paddle.linalg.norm(y, axis=1, keepdim=True)
    x = paddle.cross(y, z, axis=1)

    # local -> global
    R_local_to_global = paddle.stack([x, y, z], axis=-1)

    # obj -> global
    R_obj_to_global = paddle.bmm(R_local_to_global, R_obj_to_local)

    egocentric_quat = matrix_to_quaternion(R_obj_to_global)

    # Make sure it's unit norm.
    quat_norm = paddle.linalg.norm(egocentric_quat, axis=1, keepdim=True)
    if not paddle.allclose(quat_norm, paddle.ones_like(quat_norm), atol=1e-3):
        logger.warning(
            f"Some of the input quaternions are not unit norm: min={quat_norm.min()}, max={quat_norm.max()}; therefore normalizing."
        )
        egocentric_quat = egocentric_quat / quat_norm.clip(min=EPS)

    return egocentric_quat


def predictions_to_boxes3d(quat,
                           proj_ctr,
                           depth,
                           size,
                           locations,
                           inv_intrinsics,
                           canon_box_sizes,
                           min_depth,
                           max_depth,
                           scale_depth_by_focal_lengths_factor,
                           scale_depth_by_focal_lengths=True,
                           quat_is_allocentric=True,
                           depth_is_distance=False):
    # Normalize to make quat unit norm.
    quat = quat / paddle.linalg.norm(quat, axis=1, keepdim=True).clip(min=EPS)
    # Make sure again it's numerically unit-norm.
    quat = quat / paddle.linalg.norm(quat, axis=1, keepdim=True)

    if scale_depth_by_focal_lengths:
        pixel_size = paddle.linalg.norm(
            paddle.stack([inv_intrinsics[:, 0, 0], inv_intrinsics[:, 1, 1]],
                         axis=-1),
            axis=-1)
        depth = depth / (pixel_size * scale_depth_by_focal_lengths_factor)

    if depth_is_distance:
        depth = depth / paddle.linalg.norm(
            unproject_points2d(locations, inv_intrinsics), axis=1).clip(min=EPS)

    depth = depth.reshape([-1, 1]).clip(min_depth, max_depth)

    proj_ctr = proj_ctr + locations

    if quat_is_allocentric:
        quat = allocentric_to_egocentric(quat, proj_ctr, inv_intrinsics)

    size = (size.tanh() + 1.) * canon_box_sizes  # max size = 2 * canon_size

    return paddle.concat([quat, proj_ctr, depth, size], -1)


@manager.HEADS.add_component
class FCOS3DHead(nn.Layer):
    """
    This code is based on https://github.com/TRI-ML/dd3d/blob/main/tridet/modeling/dd3d/fcos3d.py#L55
    """

    def __init__(self,
                 in_strides,
                 in_channels,
                 num_classes=5,
                 use_scale=True,
                 depth_scale_init_factor=0.3,
                 proj_ctr_scale_init_factor=1.0,
                 use_per_level_predictors=False,
                 mean_depth_per_level=[32.594, 15.178, 8.424, 5.004, 4.662],
                 std_depth_per_level=[14.682, 7.139, 4.345, 2.399, 2.587],
                 num_convs=4,
                 use_deformable=False,
                 norm='FrozenBN',
                 class_agnostic_box3d=False,
                 per_level_predictors=False):
        super().__init__()
        self.in_strides = in_strides
        self.num_levels = len(in_strides)
        self.num_classes = num_classes
        self.use_scale = use_scale
        self.depth_scale_init_factor = depth_scale_init_factor
        self.proj_ctr_scale_init_factor = proj_ctr_scale_init_factor
        self.use_per_level_predictors = use_per_level_predictors

        self.register_buffer("mean_depth_per_level",
                             paddle.to_tensor(mean_depth_per_level))
        self.register_buffer("std_depth_per_level",
                             paddle.to_tensor(std_depth_per_level))

        assert len(
            set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        if use_deformable:
            raise ValueError("Not supported yet.")

        box3d_tower = []
        for i in range(num_convs):
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
            box3d_tower.append(
                nn.Conv2D(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=False))
            box3d_tower.append(norm_layer)
            box3d_tower.append(nn.ReLU())
        self.add_sublayer('box3d_tower', nn.Sequential(*box3d_tower))

        num_classes = self.num_classes if not class_agnostic_box3d else 1
        num_levels = self.num_levels if per_level_predictors else 1

        # 3D box branches.
        self.box3d_quat = nn.LayerList([
            nn.Conv2D(
                in_channels,
                4 * num_classes,
                kernel_size=3,
                stride=1,
                padding=1) for _ in range(num_levels)
        ])
        self.box3d_ctr = nn.LayerList([
            nn.Conv2D(
                in_channels,
                2 * num_classes,
                kernel_size=3,
                stride=1,
                padding=1) for _ in range(num_levels)
        ])
        self.box3d_depth = nn.LayerList([
            nn.Conv2D(
                in_channels,
                1 * num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=(not self.use_scale)) for _ in range(num_levels)
        ])
        self.box3d_size = nn.LayerList([
            nn.Conv2D(
                in_channels,
                3 * num_classes,
                kernel_size=3,
                stride=1,
                padding=1) for _ in range(num_levels)
        ])
        self.box3d_conf = nn.LayerList([
            nn.Conv2D(
                in_channels,
                1 * num_classes,
                kernel_size=3,
                stride=1,
                padding=1) for _ in range(num_levels)
        ])

        if self.use_scale:
            self.scales_proj_ctr = nn.LayerList([
                Scale(init_value=stride * self.proj_ctr_scale_init_factor)
                for stride in self.in_strides
            ])
            # (pre-)compute (mean, std) of depth for each level, and determine the init value here.
            self.scales_size = nn.LayerList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)])
            self.scales_conf = nn.LayerList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)])

            self.scales_depth = nn.LayerList([
                Scale(init_value=sigma * self.depth_scale_init_factor)
                for sigma in self.std_depth_per_level
            ])
            self.offsets_depth = nn.LayerList(
                [Offset(init_value=b) for b in self.mean_depth_per_level])

        self._init_weights()

    def _init_weights(self):

        for l in self.box3d_tower.sublayers():
            if isinstance(l, nn.Conv2D):
                param_init.kaiming_normal_init(
                    l.weight, mode='fan_out', nonlinearity='relu')
                if l.bias is not None:
                    param_init.constant_init(l.bias, value=0.0)

        predictors = [
            self.box3d_quat, self.box3d_ctr, self.box3d_depth, self.box3d_size,
            self.box3d_conf
        ]

        for layers in predictors:
            for l in layers.sublayers():
                if isinstance(l, nn.Conv2D):
                    param_init.kaiming_uniform_init(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        param_init.constant_init(l.bias, value=0.0)

    def forward(self, x):
        box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf = [], [], [], [], []
        dense_depth = None
        for l, features in enumerate(x):
            box3d_tower_out = self.box3d_tower(features)

            _l = l if self.use_per_level_predictors else 0

            # 3D box
            quat = self.box3d_quat[_l](box3d_tower_out)
            proj_ctr = self.box3d_ctr[_l](box3d_tower_out)
            depth = self.box3d_depth[_l](box3d_tower_out)
            size3d = self.box3d_size[_l](box3d_tower_out)
            conf3d = self.box3d_conf[_l](box3d_tower_out)

            if self.use_scale:
                # TODO: to optimize the runtime, apply this scaling in inference (and loss compute) only on FG pixels?
                proj_ctr = self.scales_proj_ctr[l](proj_ctr)
                size3d = self.scales_size[l](size3d)
                conf3d = self.scales_conf[l](conf3d)
                depth = self.offsets_depth[l](self.scales_depth[l](depth))

            box3d_quat.append(quat)
            box3d_ctr.append(proj_ctr)
            box3d_depth.append(depth)
            box3d_size.append(size3d)
            box3d_conf.append(conf3d)

        return box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth


@manager.LOSSES.add_component
class FCOS3DLoss(nn.Layer):
    """
    This code is based on https://github.com/TRI-ML/dd3d/blob/main/tridet/modeling/dd3d/fcos3d.py#L191
    """

    def __init__(
            self,
            canon_box_sizes=[
                [1.61876949, 3.89154523, 1.52969237],  # Car
                [0.62806586, 0.82038497, 1.76784787],  # Pedestrian
                [0.56898187, 1.77149234, 1.7237099],  # Cyclist
                [1.9134491, 5.15499603, 2.18998422],  # Van
                [2.61168401, 9.22692319, 3.36492722],  # Truck
                [0.5390196, 1.08098042, 1.28392158],  # Person_sitting
                [2.36044838, 15.56991038, 3.5289238],  # Tram
                [1.24489164, 2.51495357, 1.61402478],  # Misc
            ],  # (width, length, height)
            min_depth=0.1,
            max_depth=80.0,
            predict_allocentric_rot=True,
            scale_depth_by_focal_lengths=True,
            scale_depth_by_focal_lengths_factor=500.0,
            predict_distance=False,
            smooth_l1_loss_beta=0.05,
            max_loss_per_group=20.0,
            box3d_loss_weight=2.0,
            conf3d_loss_weight=1.0,
            conf_3d_temperature=1.0,
            num_classes=5,
            class_agnostic=False):
        super().__init__()
        self.box3d_reg_loss_fn = DisentangledBox3DLoss(smooth_l1_loss_beta,
                                                       max_loss_per_group)
        self.canon_box_sizes = canon_box_sizes
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.predict_allocentric_rot = predict_allocentric_rot
        self.scale_depth_by_focal_lengths = scale_depth_by_focal_lengths
        self.scale_depth_by_focal_lengths_factor = scale_depth_by_focal_lengths_factor
        self.predict_distance = predict_distance
        self.box3d_loss_weight = box3d_loss_weight
        self.conf3d_loss_weight = conf3d_loss_weight
        self.conf_3d_temperature = conf_3d_temperature
        self.class_agnostic = class_agnostic
        self.num_classes = num_classes

    def forward(self, box3d_quat, box3d_ctr, box3d_depth, box3d_size,
                box3d_conf, dense_depth, inv_intrinsics, fcos2d_info, targets):
        labels = targets['labels']
        box3d_targets = targets['box3d_targets']
        pos_inds = targets["pos_inds"]

        if pos_inds.numel() == 0:
            losses = {
                "loss_box3d_quat": box3d_quat[0].sum() * 0.,
                "loss_box3d_proj_ctr": box3d_ctr[0].sum() * 0.,
                "loss_box3d_depth": box3d_depth[0].sum() * 0.,
                "loss_box3d_size": box3d_size[0].sum() * 0.,
                "loss_conf3d": box3d_conf[0].sum() * 0.
            }
            return losses

        if len(labels) != len(box3d_targets):
            raise ValueError(
                f"The size of 'labels' and 'box3d_targets' does not match: a={len(labels)}, b={len(box3d_targets)}"
            )

        num_classes = self.num_classes if not self.class_agnostic else 1

        box3d_quat_pred = paddle.concat([
            x.transpose([0, 2, 3, 1]).reshape([-1, 4, num_classes])
            for x in box3d_quat
        ],
                                        axis=0)
        box3d_ctr_pred = paddle.concat([
            x.transpose([0, 2, 3, 1]).reshape([-1, 2, num_classes])
            for x in box3d_ctr
        ],
                                       axis=0)
        box3d_depth_pred = paddle.concat([
            x.transpose([0, 2, 3, 1]).reshape([-1, num_classes])
            for x in box3d_depth
        ],
                                         axis=0)
        box3d_size_pred = paddle.concat([
            x.transpose([0, 2, 3, 1]).reshape([-1, 3, num_classes])
            for x in box3d_size
        ],
                                        axis=0)
        box3d_conf_pred = paddle.concat([
            x.transpose([0, 2, 3, 1]).reshape([-1, num_classes])
            for x in box3d_conf
        ],
                                        axis=0)

        # 3D box disentangled loss

        if pos_inds.numel() == 1:
            box3d_targets = box3d_targets[pos_inds].unsqueeze(0)

            box3d_quat_pred = box3d_quat_pred[pos_inds].unsqueeze(0)
            box3d_ctr_pred = box3d_ctr_pred[pos_inds].unsqueeze(0)
            box3d_depth_pred = box3d_depth_pred[pos_inds].unsqueeze(0)
            box3d_size_pred = box3d_size_pred[pos_inds].unsqueeze(0)
            box3d_conf_pred = box3d_conf_pred[pos_inds].unsqueeze(0)
        else:
            box3d_targets = box3d_targets[pos_inds]

            box3d_quat_pred = box3d_quat_pred[pos_inds]
            box3d_ctr_pred = box3d_ctr_pred[pos_inds]
            box3d_depth_pred = box3d_depth_pred[pos_inds]
            box3d_size_pred = box3d_size_pred[pos_inds]
            box3d_conf_pred = box3d_conf_pred[pos_inds]

        if self.class_agnostic:
            box3d_quat_pred = box3d_quat_pred.squeeze(-1)
            box3d_ctr_pred = box3d_ctr_pred.squeeze(-1)
            box3d_depth_pred = box3d_depth_pred.squeeze(-1)
            box3d_size_pred = box3d_size_pred.squeeze(-1)
            box3d_conf_pred = box3d_conf_pred.squeeze(-1)
        else:
            I = labels[pos_inds][..., None, None]
            box3d_quat_pred = paddle.take_along_axis(
                box3d_quat_pred, indices=I.tile([1, 4, 1]), axis=2).squeeze(-1)
            box3d_ctr_pred = paddle.take_along_axis(
                box3d_ctr_pred, indices=I.tile([1, 2, 1]), axis=2).squeeze(-1)
            box3d_depth_pred = paddle.take_along_axis(
                box3d_depth_pred, indices=I.squeeze(-1), axis=1).squeeze(-1)
            box3d_size_pred = paddle.take_along_axis(
                box3d_size_pred, indices=I.tile([1, 3, 1]), axis=2).squeeze(-1)
            box3d_conf_pred = paddle.take_along_axis(
                box3d_conf_pred, indices=I.squeeze(-1), axis=1).squeeze(-1)

        canon_box_sizes = paddle.to_tensor(
            self.canon_box_sizes)[labels[pos_inds]]

        locations = targets["locations"][pos_inds]
        im_inds = targets["im_inds"][pos_inds]
        inv_intrinsics = inv_intrinsics[im_inds]
        if im_inds.numel() == 1:
            inv_intrinsics = inv_intrinsics.unsqueeze(0)

        box3d_pred = predictions_to_boxes3d(
            box3d_quat_pred,
            box3d_ctr_pred,
            box3d_depth_pred,
            box3d_size_pred,
            locations,
            inv_intrinsics,
            canon_box_sizes,
            self.min_depth,
            self.max_depth,
            scale_depth_by_focal_lengths_factor=self.
            scale_depth_by_focal_lengths_factor,
            scale_depth_by_focal_lengths=self.scale_depth_by_focal_lengths,
            quat_is_allocentric=self.predict_allocentric_rot,
            depth_is_distance=self.predict_distance)

        centerness_targets = fcos2d_info["centerness_targets"]
        loss_denom = fcos2d_info["loss_denom"]

        losses_box3d, box3d_l1_error = self.box3d_reg_loss_fn(
            box3d_pred, box3d_targets, locations, inv_intrinsics,
            centerness_targets)

        losses_box3d = {
            k: self.box3d_loss_weight * v / loss_denom
            for k, v in losses_box3d.items()
        }

        conf_3d_targets = paddle.exp(
            -1. / self.conf_3d_temperature * box3d_l1_error)
        loss_conf3d = F.binary_cross_entropy_with_logits(
            box3d_conf_pred, conf_3d_targets, reduction='none')
        loss_conf3d = self.conf3d_loss_weight * (
            loss_conf3d * centerness_targets).sum() / loss_denom

        losses = {"loss_conf3d": loss_conf3d, **losses_box3d}

        return losses


@manager.MODELS.add_component
class FCOS3DInference(nn.Layer):
    """
    This code is based on https://github.com/TRI-ML/dd3d/blob/main/tridet/modeling/dd3d/fcos3d.py#L302
    """

    def __init__(
            self,
            canon_box_sizes=[
                [1.61876949, 3.89154523, 1.52969237],  # Car
                [0.62806586, 0.82038497, 1.76784787],  # Pedestrian
                [0.56898187, 1.77149234, 1.7237099],  # Cyclist
                [1.9134491, 5.15499603, 2.18998422],  # Van
                [2.61168401, 9.22692319, 3.36492722],  # Truck
                [0.5390196, 1.08098042, 1.28392158],  # Person_sitting
                [2.36044838, 15.56991038, 3.5289238],  # Tram
                [1.24489164, 2.51495357, 1.61402478],  # Misc
            ],  # (width, length, height)
            min_depth=0.1,
            max_depth=80.0,
            predict_allocentric_rot=True,
            scale_depth_by_focal_lengths=True,
            scale_depth_by_focal_lengths_factor=500.0,
            predict_distance=False,
            num_classes=5,
            class_agnostic=False):
        super().__init__()
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.predict_distance = predict_distance
        self.canon_box_sizes = canon_box_sizes
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.predict_allocentric_rot = predict_allocentric_rot
        self.scale_depth_by_focal_lengths_factor = scale_depth_by_focal_lengths_factor
        self.scale_depth_by_focal_lengths = scale_depth_by_focal_lengths

    def forward(self, box3d_quat, box3d_ctr, box3d_depth, box3d_size,
                box3d_conf, inv_intrinsics, pred_instances, fcos2d_info):
        # pred_instances: # List[List[Instances]], shape = (L, B)
        for lvl, (box3d_quat_lvl, box3d_ctr_lvl, box3d_depth_lvl, box3d_size_lvl, box3d_conf_lvl) in \
            enumerate(zip(box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf)):

            # In-place modification: update per-level pred_instances.
            self.forward_for_single_feature_map(
                box3d_quat_lvl, box3d_ctr_lvl, box3d_depth_lvl, box3d_size_lvl,
                box3d_conf_lvl, inv_intrinsics, pred_instances[lvl],
                fcos2d_info[lvl])  # List of Instances; one for each image.

    def forward_for_single_feature_map(self, box3d_quat, box3d_ctr, box3d_depth,
                                       box3d_size, box3d_conf, inv_intrinsics,
                                       pred_instances, fcos2d_info):
        N = box3d_quat.shape[0]

        num_classes = self.num_classes if not self.class_agnostic else 1

        box3d_quat = box3d_quat.transpose([0, 2, 3,
                                           1]).reshape([N, -1, 4, num_classes])
        box3d_ctr = box3d_ctr.transpose([0, 2, 3,
                                         1]).reshape([N, -1, 2, num_classes])
        box3d_depth = box3d_depth.transpose([0, 2, 3,
                                             1]).reshape([N, -1, num_classes])
        box3d_size = box3d_size.transpose([0, 2, 3,
                                           1]).reshape([N, -1, 3, num_classes])
        box3d_conf = F.sigmoid(
            box3d_conf.transpose([0, 2, 3, 1]).reshape([N, -1, num_classes]))

        for i in range(N):
            fg_inds_per_im = fcos2d_info['fg_inds_per_im'][i]
            class_inds_per_im = fcos2d_info['class_inds_per_im'][i]
            topk_indices = fcos2d_info['topk_indices'][i]

            if fg_inds_per_im.shape[0] == 0:
                box3d_conf_per_im = paddle.zeros([0, num_classes])
                pred_instances[i]['pred_boxes3d'] = paddle.zeros([0, 10])
            else:
                if fg_inds_per_im.shape[0] == 1:
                    box3d_quat_per_im = box3d_quat[i][fg_inds_per_im].unsqueeze(
                        0)
                    box3d_ctr_per_im = box3d_ctr[i][fg_inds_per_im].unsqueeze(0)
                    box3d_depth_per_im = box3d_depth[i][
                        fg_inds_per_im].unsqueeze(0)
                    box3d_size_per_im = box3d_size[i][fg_inds_per_im].unsqueeze(
                        0)
                    box3d_conf_per_im = box3d_conf[i][fg_inds_per_im].unsqueeze(
                        0)
                else:
                    box3d_quat_per_im = box3d_quat[i][fg_inds_per_im]
                    box3d_ctr_per_im = box3d_ctr[i][fg_inds_per_im]
                    box3d_depth_per_im = box3d_depth[i][fg_inds_per_im]
                    box3d_size_per_im = box3d_size[i][fg_inds_per_im]
                    box3d_conf_per_im = box3d_conf[i][fg_inds_per_im]

                if self.class_agnostic:
                    box3d_quat_per_im = box3d_quat_per_im.squeeze(-1)
                    box3d_ctr_per_im = box3d_ctr_per_im.squeeze(-1)
                    box3d_depth_per_im = box3d_depth_per_im.squeeze(-1)
                    box3d_size_per_im = box3d_size_per_im.squeeze(-1)
                    box3d_conf_per_im = box3d_conf_per_im.squeeze(-1)
                else:
                    I = class_inds_per_im[..., None, None]
                    box3d_quat_per_im = paddle.take_along_axis(
                        box3d_quat_per_im, indices=I.tile([1, 4, 1]),
                        axis=2).squeeze(-1)
                    box3d_ctr_per_im = paddle.take_along_axis(
                        box3d_ctr_per_im, indices=I.tile([1, 2, 1]),
                        axis=2).squeeze(-1)
                    box3d_depth_per_im = paddle.take_along_axis(
                        box3d_depth_per_im, indices=I.squeeze(-1),
                        axis=1).squeeze(-1)
                    box3d_size_per_im = paddle.take_along_axis(
                        box3d_size_per_im, indices=I.tile([1, 3, 1]),
                        axis=2).squeeze(-1)
                    box3d_conf_per_im = paddle.take_along_axis(
                        box3d_conf_per_im, indices=I.squeeze(-1),
                        axis=1).squeeze(-1)

                if topk_indices is not None:
                    box3d_quat_per_im = box3d_quat_per_im[topk_indices]
                    box3d_ctr_per_im = box3d_ctr_per_im[topk_indices]
                    box3d_depth_per_im = box3d_depth_per_im[topk_indices]
                    box3d_size_per_im = box3d_size_per_im[topk_indices]
                    box3d_conf_per_im = box3d_conf_per_im[topk_indices]

                canon_box_sizes = paddle.to_tensor(
                    self.canon_box_sizes)[pred_instances[i]['pred_classes']]
                inv_K = inv_intrinsics[i][None, ...].expand(
                    [len(box3d_quat_per_im), 3, 3])
                locations = pred_instances[i]['locations']
                pred_boxes3d = predictions_to_boxes3d(
                    box3d_quat_per_im,
                    box3d_ctr_per_im,
                    box3d_depth_per_im,
                    box3d_size_per_im,
                    locations,
                    inv_K,
                    canon_box_sizes,
                    self.min_depth,
                    self.max_depth,
                    scale_depth_by_focal_lengths_factor=self.
                    scale_depth_by_focal_lengths_factor,
                    scale_depth_by_focal_lengths=self.
                    scale_depth_by_focal_lengths,
                    quat_is_allocentric=self.predict_allocentric_rot,
                    depth_is_distance=self.predict_distance)

                pred_instances[i]['pred_boxes3d'] = pred_boxes3d

            # scores_per_im = pred_instances[i].scores.square()
            # NOTE: Before refactoring, the squared score was used. Is raw 2D score better?
            scores_per_im = pred_instances[i]['scores']
            if scores_per_im.shape[0] == 0:
                scores_3d_per_im = paddle.zeros(scores_per_im.shape)
            else:
                scores_3d_per_im = scores_per_im * box3d_conf_per_im

            # In-place modification: add fields to instances.
            pred_instances[i]['scores_3d'] = scores_3d_per_im
