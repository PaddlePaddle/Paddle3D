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

# This code is based on https://github.com/yifanzhang713/IA-SSD/blob/main/pcdet/models/dense_heads/IASSD_head.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager

from .iassd_coder import PointResidual_BinOri_Coder
from .iassd_loss import (WeightedClassificationLoss, WeightedSmoothL1Loss,
                         get_corner_loss_lidar)
from paddle3d.ops import roiaware_pool3d
from paddle3d.models.common import enlarge_box3d, rotate_points_along_z

__all__ = ["IASSD_Head"]


@manager.HEADS.add_component
class IASSD_Head(nn.Layer):
    """Head of IA-SSD

    Args:
        input_channle (int): input feature channel of IA-SSD Head.
        cls_fc (List[int]): hidden dim of box classification branch.
        reg_fc (List[int]): hidden dim of box regression branch.
        num_classes (int): number of classes.
        target_config (dict): config of box coder to encode boxes.
        loss_config (dict): config of loss computation.
    """

    def __init__(self, input_channel, cls_fc, reg_fc, num_classes,
                 target_config, loss_config):
        super().__init__()

        self.forward_ret_dict = None
        self.num_classes = num_classes
        self.target_config = target_config
        self.loss_config = loss_config
        self.box_coder = PointResidual_BinOri_Coder(
            **target_config.get("box_coder_config"))

        self.cls_center_layers = self.make_fc_layers(
            fc_cfg=cls_fc,
            input_channel=input_channel,
            output_channel=num_classes)
        self.box_center_layers = self.make_fc_layers(
            fc_cfg=reg_fc,
            input_channel=input_channel,
            output_channel=self.box_coder.code_size,
        )
        self.build_loss()

    def make_fc_layers(self, fc_cfg, input_channel, output_channel):
        fc_layers = []
        for k in range(len(fc_cfg)):
            fc_layers.extend([
                nn.Linear(input_channel, fc_cfg[k], bias_attr=False),
                nn.BatchNorm1D(fc_cfg[k]),
                nn.ReLU(),
            ])
            input_channel = fc_cfg[k]
        fc_layers.append(
            nn.Linear(input_channel, output_channel, bias_attr=True))
        return nn.Sequential(*fc_layers)

    def build_loss(self):
        # classification loss
        if self.loss_config["loss_cls"] == "WeightedClassificationLoss":
            self.add_sublayer("cls_loss_func", WeightedClassificationLoss())
        else:
            raise NotImplementedError

        # regression loss
        if self.loss_config["loss_reg"] == "WeightedSmoothL1Loss":
            self.add_sublayer(
                "reg_loss_func",
                WeightedSmoothL1Loss(
                    code_weights=paddle.to_tensor(
                        self.loss_config["loss_weight"]["code_weights"])),
            )
        else:
            raise NotImplementedError

        # instance-aware loss
        if self.loss_config["loss_ins"] == "WeightedClassificationLoss":
            self.add_sublayer("ins_loss_func", WeightedClassificationLoss())
        else:
            raise NotImplementedError

    def get_loss(self):
        # vote loss
        center_loss_reg = self.get_contextual_vote_loss()

        # semantic loss in SA
        sa_loss_cls = self.get_sa_ins_layer_loss()

        # cls loss
        center_loss_cls = self.get_center_cls_layer_loss()

        # reg loss
        center_loss_box = self.get_center_box_binori_layer_loss()

        # corner loss
        if self.loss_config.get("corner_loss_regularization", False):
            corner_loss = self.get_corner_layer_loss()

        point_loss = (center_loss_reg + center_loss_cls + center_loss_box +
                      corner_loss + sa_loss_cls)
        return point_loss

    def get_contextual_vote_loss(self):
        pos_mask = self.forward_ret_dict["center_origin_cls_labels"] > 0
        center_origin_loss_box = []
        for i in self.forward_ret_dict["center_origin_cls_labels"].unique():
            if i <= 0:
                continue
            simple_pos_mask = self.forward_ret_dict[
                "center_origin_cls_labels"] == i
            center_box_labels = self.forward_ret_dict[
                "center_origin_gt_box_of_fg_points"][:, 0:3][(
                    pos_mask & simple_pos_mask)[pos_mask == 1]]
            centers_origin = self.forward_ret_dict["centers_origin"]
            ctr_offsets = self.forward_ret_dict["ctr_offsets"]
            centers_pred = centers_origin + ctr_offsets
            centers_pred = centers_pred[simple_pos_mask][:, 1:4]
            simple_center_origin_loss_box = F.smooth_l1_loss(
                centers_pred, center_box_labels)
            center_origin_loss_box.append(
                simple_center_origin_loss_box.unsqueeze(-1))
        center_origin_loss_box = paddle.concat(
            center_origin_loss_box, axis=-1).mean()
        center_origin_loss_box = (
            center_origin_loss_box *
            self.loss_config["loss_weight"]["vote_weight"])

        return center_origin_loss_box

    def get_center_cls_layer_loss(self):
        point_cls_labels = self.forward_ret_dict["center_cls_labels"].reshape(
            [-1])
        point_cls_preds = self.forward_ret_dict["center_cls_preds"].reshape(
            [-1, self.num_classes])
        positives = point_cls_labels > 0
        negative_cls_weights = (point_cls_labels == 0) * 1.0

        cls_weights = (
            1.0 * negative_cls_weights + 1.0 * positives).astype("float32")
        pos_normalizer = positives.sum(axis=0).astype("float32")
        cls_weights /= paddle.clip(pos_normalizer, min=1.0)

        one_hot_targets = F.one_hot(
            (point_cls_labels *
             (point_cls_labels >= 0).astype("int64")).astype("int64"),
            self.num_classes + 1,
        )
        one_hot_targets = one_hot_targets[..., 1:]

        if self.loss_config.get("centerness_regularization", False):
            centerness_mask = self.generate_center_ness_mask()
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(
                -1).expand(one_hot_targets.shape)

        point_loss_cls = (self.cls_loss_func(
            point_cls_preds, one_hot_targets,
            weights=cls_weights).mean(axis=-1).sum())
        point_loss_cls = (point_loss_cls *
                          self.loss_config["loss_weight"]["point_cls_weight"])

        return point_loss_cls

    def get_sa_ins_layer_loss(self):
        sa_ins_labels = self.forward_ret_dict["sa_ins_labels"]
        sa_ins_preds = self.forward_ret_dict["sa_ins_preds"]
        sa_centerness_mask = self.generate_sa_center_ness_mask()
        sa_ins_loss, ignore = 0, 0
        for i in range(len(sa_ins_labels)):  # valid when i=1,2 for IA-SSD
            if len(sa_ins_preds[i]) != 0:
                point_cls_preds = sa_ins_preds[i][..., 1:].reshape(
                    [-1, self.num_classes])
            else:
                ignore += 1
                continue
            point_cls_labels = sa_ins_labels[i].reshape([-1])
            positives = point_cls_labels > 0
            negative_cls_weights = (point_cls_labels == 0) * 1.0
            cls_weights = (
                negative_cls_weights + 1.0 * positives).astype("float32")
            pos_normalizer = positives.sum(axis=0).astype("float32")
            cls_weights /= paddle.clip(pos_normalizer, min=1.0)

            one_hot_targets = F.one_hot(
                (point_cls_labels *
                 (point_cls_labels >= 0).astype("int64")).astype("int64"),
                self.num_classes + 1,
            )
            one_hot_targets = one_hot_targets[..., 1:]

            if "ctr" in self.loss_config["sample_method_list"][i + 1]:
                centerness_mask = sa_centerness_mask[i]
                one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(
                    -1).expand(one_hot_targets.shape)

            point_loss_ins = (self.ins_loss_func(
                point_cls_preds, one_hot_targets,
                weights=cls_weights).mean(axis=-1).sum())
            loss_weights = self.loss_config["loss_weight"]["ins_aware_weight"]
            point_loss_ins = point_loss_ins * loss_weights[i]
            sa_ins_loss += point_loss_ins
        sa_ins_loss = sa_ins_loss / (len(sa_ins_labels) - ignore)

        return sa_ins_loss

    def generate_center_ness_mask(self):
        pos_mask = self.forward_ret_dict["center_cls_labels"] > 0
        gt_boxes = self.forward_ret_dict["center_gt_box_of_fg_points"]
        centers = self.forward_ret_dict["centers"][:, 1:]
        centers = centers[pos_mask].clone().detach()
        offset_xyz = centers[:, 0:3] - gt_boxes[:, 0:3]

        offset_xyz_canical = rotate_points_along_z(
            offset_xyz.unsqueeze(axis=1), -gt_boxes[:, 6]).squeeze(axis=1)

        template = paddle.to_tensor([[1, 1, 1], [-1, -1, -1]],
                                    dtype=gt_boxes.dtype) / 2
        margin = (gt_boxes[:, None, 3:6].expand([gt_boxes.shape[0], 2, 3]) *
                  template[None, :, :])
        distance = margin - offset_xyz_canical[:, None, :].expand(
            [offset_xyz_canical.shape[0], 2, offset_xyz_canical.shape[1]])
        distance[:, 1, :] = -1 * distance[:, 1, :]
        distance_min = paddle.where(distance[:, 0, :] < distance[:, 1, :],
                                    distance[:, 0, :], distance[:, 1, :])
        distance_max = paddle.where(distance[:, 0, :] > distance[:, 1, :],
                                    distance[:, 0, :], distance[:, 1, :])

        centerness = distance_min / distance_max
        centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
        centerness = paddle.clip(centerness, min=1e-6)
        centerness = paddle.pow(centerness, 1 / 3)

        centerness_mask = paddle.zeros(pos_mask.shape).astype("float32")
        centerness_mask[pos_mask] = centerness

        return centerness_mask

    def generate_sa_center_ness_mask(self):
        sa_pos_mask = self.forward_ret_dict["sa_ins_labels"]
        sa_gt_boxes = self.forward_ret_dict["sa_gt_box_of_fg_points"]
        sa_xyz_coords = self.forward_ret_dict["sa_xyz_coords"]
        sa_centerness_mask = []
        for i in range(len(sa_pos_mask)):
            pos_mask = sa_pos_mask[i] > 0
            gt_boxes = sa_gt_boxes[i]
            xyz_coords = sa_xyz_coords[i].reshape(
                [-1, sa_xyz_coords[i].shape[-1]])[:, 1:]
            xyz_coords = xyz_coords[pos_mask].clone().detach()
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3]
            offset_xyz_canical = rotate_points_along_z(
                offset_xyz.unsqueeze(axis=1), -gt_boxes[:, 6]).squeeze(axis=1)

            template = (paddle.to_tensor([[1, 1, 1], [-1, -1, -1]],
                                         dtype=gt_boxes.dtype) / 2)
            margin = (gt_boxes[:, None, 3:6].expand([gt_boxes.shape[0], 2, 3]) *
                      template[None, :, :])
            distance = margin - offset_xyz_canical[:, None, :].expand(
                [offset_xyz_canical.shape[0], 2, offset_xyz_canical.shape[1]])
            distance[:, 1, :] = -1 * distance[:, 1, :]
            distance_min = paddle.where(
                distance[:, 0, :] < distance[:, 1, :],
                distance[:, 0, :],
                distance[:, 1, :],
            )
            distance_max = paddle.where(
                distance[:, 0, :] > distance[:, 1, :],
                distance[:, 0, :],
                distance[:, 1, :],
            )

            centerness = distance_min / distance_max
            centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
            centerness = paddle.clip(centerness, min=1e-6)
            centerness = paddle.pow(centerness, 1 / 3)

            centerness_mask = paddle.zeros(pos_mask.shape).astype("float32")
            centerness_mask[pos_mask] = centerness

            sa_centerness_mask.append(centerness_mask)

        return sa_centerness_mask

    def get_center_box_binori_layer_loss(self):
        pos_mask = self.forward_ret_dict["center_cls_labels"] > 0
        point_box_labels = self.forward_ret_dict["center_box_labels"]
        point_box_preds = self.forward_ret_dict["center_box_preds"]

        reg_weights = pos_mask.astype("float32")
        pos_normalizer = pos_mask.sum().astype("float32")
        reg_weights /= paddle.clip(pos_normalizer, min=1.0)

        pred_box_xyzwhl = point_box_preds[:, :6]
        label_box_xyzwhl = point_box_labels[:, :6]

        point_loss_box_src = self.reg_loss_func(
            pred_box_xyzwhl[None, ...],
            label_box_xyzwhl[None, ...],
            weights=reg_weights[None, ...],
        )
        point_loss_xyzwhl = point_loss_box_src.sum()

        pred_ori_bin_id = point_box_preds[:, 6:6 + self.box_coder.bin_size]
        pred_ori_bin_res = point_box_preds[:, 6 + self.box_coder.bin_size:]

        label_ori_bin_id = point_box_labels[:, 6]
        label_ori_bin_res = point_box_labels[:, 7]
        criterion = nn.CrossEntropyLoss(reduction="none")
        loss_ori_cls = criterion(pred_ori_bin_id,
                                 label_ori_bin_id.astype("int64"))
        loss_ori_cls = paddle.sum(loss_ori_cls * reg_weights)

        label_id_one_hot = F.one_hot(
            label_ori_bin_id.astype("int64"), self.box_coder.bin_size)
        pred_ori_bin_res = paddle.sum(
            pred_ori_bin_res * label_id_one_hot.astype("float32"), axis=-1)
        loss_ori_reg = F.smooth_l1_loss(pred_ori_bin_res, label_ori_bin_res)
        loss_ori_reg = paddle.sum(loss_ori_reg * reg_weights)

        loss_ori_cls = loss_ori_cls * self.loss_config["loss_weight"][
            "dir_weight"]
        point_loss_box = point_loss_xyzwhl + loss_ori_reg + loss_ori_cls
        point_loss_box = (point_loss_box *
                          self.loss_config["loss_weight"]["point_box_weight"])

        return point_loss_box

    def get_corner_layer_loss(self):
        pos_mask = self.forward_ret_dict["center_cls_labels"] > 0
        gt_boxes = self.forward_ret_dict["center_gt_box_of_fg_points"]
        pred_boxes = self.forward_ret_dict["point_box_preds"]
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = get_corner_loss_lidar(pred_boxes[:, 0:7],
                                            gt_boxes[:, 0:7])
        loss_corner = loss_corner.mean()
        loss_corner = loss_corner * self.loss_config["loss_weight"][
            "corner_weight"]
        return loss_corner

    def assign_stack_targets_IASSD(
            self,
            points,
            gt_boxes,
            extend_gt_boxes=None,
            weighted_labels=False,
            ret_box_labels=False,
            ret_offset_labels=True,
            set_ignore_flag=True,
            use_ball_constraint=False,
            central_radius=2.0,
            use_query_assign=False,
            central_radii=2.0,
            use_ex_gt_assign=False,
            fg_pc_ignore=False,
            binary_label=False,
    ):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)
        """
        assert len(
            points.shape
        ) == 2 and points.shape[1] == 4, "points.shape=%s" % str(points.shape)
        assert (len(gt_boxes.shape) == 3
                and gt_boxes.shape[2] == 8), "gt_boxes.shape=%s" % str(
                    gt_boxes.shape)
        assert (extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3
                and extend_gt_boxes.shape[2] == 8
                ), "extend_gt_boxes.shape=%s" % str(extend_gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = paddle.zeros([points.shape[0]]).astype("int64")
        point_box_labels = (paddle.zeros([points.shape[0], 8])
                            if ret_box_labels else None)
        box_idxs_labels = paddle.zeros([points.shape[0]]).astype("int64")
        gt_boxes_of_fg_points = []
        gt_box_of_points = paddle.zeros([points.shape[0], 8],
                                        dtype=gt_boxes.dtype)

        for k in range(batch_size):
            bs_mask = bs_idx == k
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = paddle.zeros([bs_mask.sum()],
                                                   dtype=point_cls_labels.dtype)
            box_idxs_of_pts = (roiaware_pool3d.points_in_boxes_gpu(
                points_single.unsqueeze(axis=0),
                gt_boxes[k:k + 1, :, 0:7]).astype("int64").squeeze(axis=0))
            box_fg_flag = box_idxs_of_pts >= 0

            if use_ex_gt_assign:
                extend_box_idxs_of_pts = (roiaware_pool3d.points_in_boxes_gpu(
                    points_single.unsqueeze(axis=0),
                    extend_gt_boxes[k:k + 1, :, 0:7],
                ).astype("int64").squeeze(axis=0))
                extend_fg_flag = extend_box_idxs_of_pts >= 0

                extend_box_idxs_of_pts[box_fg_flag] = box_idxs_of_pts[
                    box_fg_flag]  # instance points should keep unchanged

                if fg_pc_ignore:
                    fg_flag = extend_fg_flag ^ box_fg_flag
                    extend_box_idxs_of_pts[box_idxs_of_pts != -1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = extend_fg_flag
                    box_idxs_of_pts = extend_box_idxs_of_pts

            elif set_ignore_flag:
                extend_box_idxs_of_pts = (roiaware_pool3d.points_in_boxes_gpu(
                    points_single.unsqueeze(axis=0),
                    extend_gt_boxes[k:k + 1, :, 0:7],
                ).astype("int64").squeeze(axis=0))
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = (box_centers - points_single).norm(
                    axis=1) < central_radius
                fg_flag = box_fg_flag & ball_flag

            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = (
                1 if self.num_classes == 1 or binary_label else
                gt_box_of_fg_points[:, -1].astype("int64"))
            point_cls_labels[bs_mask] = point_cls_labels_single
            bg_flag = point_cls_labels_single == 0  # except ignore_id
            # box_bg_flag
            fg_flag = fg_flag ^ (fg_flag & bg_flag)
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]

            gt_boxes_of_fg_points.append(gt_box_of_fg_points)
            box_idxs_labels[bs_mask] = box_idxs_of_pts
            # FIXME: -1 index slice is not supported in paddle
            box_idxs_of_pts[box_idxs_of_pts == -1] = gt_boxes[k].shape[0] - 1
            gt_box_of_points[bs_mask] = gt_boxes[k][box_idxs_of_pts]

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = paddle.zeros(
                    [bs_mask.sum(), 8], dtype=point_box_labels.dtype)
                fg_point_box_labels = self.box_coder.encode_paddle(
                    gt_boxes=gt_box_of_fg_points[:, :-1],
                    points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].astype("int64"),
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

        gt_boxes_of_fg_points = paddle.concat(gt_boxes_of_fg_points, axis=0)
        targets_dict = {
            "point_cls_labels": point_cls_labels,
            "point_box_labels": point_box_labels,
            "gt_box_of_fg_points": gt_boxes_of_fg_points,
            "box_idxs_labels": box_idxs_labels,
            "gt_box_of_points": gt_box_of_points,
        }
        return targets_dict

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                batch_size: int
                centers: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                centers_origin: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                encoder_coords: List of point_coords in SA
                bboxes_3d (optional): (B, M, 8)
        Returns:
            target_dict:
            ...
        """
        gt_boxes = input_dict["bboxes_3d"]
        batch_size = input_dict["batch_size"]
        targets_dict = {}
        extend_gt = gt_boxes

        extend_gt_boxes = enlarge_box3d(
            extend_gt.reshape([-1, extend_gt.shape[-1]]),
            extra_width=self.target_config.get("gt_extra_width"),
        ).reshape([batch_size, -1, extend_gt.shape[-1]])
        assert gt_boxes.shape.__len__() == 3, "bboxes_3d.shape=%s" % str(
            gt_boxes.shape)
        center_targets_dict = self.assign_stack_targets_IASSD(
            points=input_dict["centers"].detach(),
            gt_boxes=extend_gt,
            extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True,
            use_ball_constraint=False,
            ret_box_labels=True,
        )

        targets_dict["center_gt_box_of_fg_points"] = center_targets_dict[
            "gt_box_of_fg_points"]
        targets_dict["center_cls_labels"] = center_targets_dict[
            "point_cls_labels"]
        targets_dict["center_box_labels"] = center_targets_dict[
            "point_box_labels"]
        targets_dict["center_gt_box_of_points"] = center_targets_dict[
            "gt_box_of_points"]

        (
            sa_ins_labels,
            sa_gt_box_of_fg_points,
            sa_xyz_coords,
            sa_gt_box_of_points,
            sa_box_idxs_labels,
        ) = ([], [], [], [], [])
        sa_ins_preds = input_dict["sa_ins_preds"]
        for i in range(1, len(sa_ins_preds)):  # valid when i = 1,2 for IA-SSD
            sa_xyz = input_dict["encoder_coords"][i]
            if i == 1:
                extend_gt_boxes = enlarge_box3d(
                    gt_boxes.reshape([-1, gt_boxes.shape[-1]]),
                    extra_width=[0.5, 0.5, 0.5],
                ).reshape([batch_size, -1, gt_boxes.shape[-1]])
                sa_targets_dict = self.assign_stack_targets_IASSD(
                    points=sa_xyz.reshape([-1, sa_xyz.shape[-1]]).detach(),
                    gt_boxes=gt_boxes,
                    extend_gt_boxes=extend_gt_boxes,
                    set_ignore_flag=True,
                    use_ex_gt_assign=False,
                )
            if i >= 2:
                extend_gt_boxes = enlarge_box3d(
                    gt_boxes.reshape([-1, gt_boxes.shape[-1]]),
                    extra_width=[0.5, 0.5, 0.5],
                ).reshape([batch_size, -1, gt_boxes.shape[-1]])
                sa_targets_dict = self.assign_stack_targets_IASSD(
                    points=sa_xyz.reshape([-1, sa_xyz.shape[-1]]).detach(),
                    gt_boxes=gt_boxes,
                    extend_gt_boxes=extend_gt_boxes,
                    set_ignore_flag=False,
                    use_ex_gt_assign=True,
                )
            sa_xyz_coords.append(sa_xyz)
            sa_ins_labels.append(sa_targets_dict["point_cls_labels"])
            sa_gt_box_of_fg_points.append(
                sa_targets_dict["gt_box_of_fg_points"])
            sa_gt_box_of_points.append(sa_targets_dict["gt_box_of_points"])
            sa_box_idxs_labels.append(sa_targets_dict["box_idxs_labels"])

        targets_dict["sa_ins_labels"] = sa_ins_labels
        targets_dict["sa_gt_box_of_fg_points"] = sa_gt_box_of_fg_points
        targets_dict["sa_xyz_coords"] = sa_xyz_coords
        targets_dict["sa_gt_box_of_points"] = sa_gt_box_of_points
        targets_dict["sa_box_idxs_labels"] = sa_box_idxs_labels

        extend_gt_boxes = enlarge_box3d(
            gt_boxes.reshape([-1, gt_boxes.shape[-1]]),
            extra_width=self.target_config.get("extra_width"),
        ).reshape([batch_size, -1, gt_boxes.shape[-1]])

        points = input_dict["centers_origin"].detach()

        center_targets_dict = self.assign_stack_targets_IASSD(
            points=points,
            gt_boxes=gt_boxes,
            extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True,
            use_ball_constraint=False,
            ret_box_labels=True,
            use_ex_gt_assign=True,
            fg_pc_ignore=False,
        )
        targets_dict["center_origin_gt_box_of_fg_points"] = center_targets_dict[
            "gt_box_of_fg_points"]
        targets_dict["center_origin_cls_labels"] = center_targets_dict[
            "point_cls_labels"]
        targets_dict["center_origin_box_idxs_of_pts"] = center_targets_dict[
            "box_idxs_labels"]
        targets_dict["gt_box_of_center_origin"] = center_targets_dict[
            "gt_box_of_points"]

        return targets_dict

    def generate_predicted_boxes(self, points, point_cls_preds,
                                 point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        """
        pred_classes = point_cls_preds.argmax(axis=-1)
        point_box_preds = self.box_coder.decode_paddle(point_box_preds, points,
                                                       pred_classes + 1)

        return point_cls_preds, point_box_preds

    def forward(self, batch_dict):
        center_features = batch_dict[
            "centers_features"]  # (total_centers, C) total_centers = bs * npoints
        center_coords = batch_dict["centers"]  # (total_centers, 4)
        center_cls_preds = self.cls_center_layers(
            center_features)  # (total_centers, num_class)
        center_box_preds = self.box_center_layers(
            center_features)  # (total_centers, box_code_size)

        ret_dict = {
            "center_cls_preds": center_cls_preds,
            "center_box_preds": center_box_preds,
            "ctr_offsets": batch_dict["ctr_offsets"],
            "centers": batch_dict["centers"],
            "centers_origin": batch_dict["centers_origin"],
            "sa_ins_preds": batch_dict["sa_ins_preds"],
        }

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict.update(targets_dict)

        if (not self.training or self.loss_config["corner_loss_regularization"]
                or self.loss_config["centerness_regularization"]):

            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=center_coords[:, 1:4],
                point_cls_preds=center_cls_preds,
                point_box_preds=center_box_preds,
            )

            batch_dict["batch_cls_preds"] = point_cls_preds
            batch_dict["batch_box_preds"] = point_box_preds
            batch_dict["batch_index"] = center_coords[:, 0]
            batch_dict["cls_preds_normalized"] = False

            ret_dict["point_box_preds"] = point_box_preds

        self.forward_ret_dict = ret_dict

        return batch_dict
