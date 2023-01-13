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

# This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/3f992837ad659f050df38d7b0978372425be16ff/mmdet3d/models/dense_heads/train_mixins.py

from functools import partial

import numpy as np
import paddle
from six.moves import map, zip


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (paddle.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        paddle.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - paddle.floor(val / period + offset) * period


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = paddle.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class AnchorTrainMixin(object):
    """Mixin class for target assigning of dense heads."""

    def anchor_target_3d(self,
                         anchor_list,
                         gt_bboxes_list,
                         input_metas,
                         gt_bboxes_ignore_list=None,
                         gt_labels_list=None,
                         label_channels=1,
                         num_classes=1,
                         sampling=True):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            gt_bboxes_list (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each image.
            input_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (None | list): Ignore list of gt bboxes.
            gt_labels_list (list[paddle.Tensor]): Gt labels of batches.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple (list, list, list, list, list, list, int, int):
                Anchor targets, including labels, label weights,
                bbox targets, bbox weights, direction targets,
                direction weights, number of postive anchors and
                number of negative anchors.
        """
        num_imgs = len(input_metas)
        assert len(anchor_list) == num_imgs

        if isinstance(anchor_list[0][0], list):
            # sizes of anchors are different
            # anchor number of a single level
            num_level_anchors = [
                sum([anchor.size(0) for anchor in anchors])
                for anchors in anchor_list[0]
            ]
            for i in range(num_imgs):
                anchor_list[i] = anchor_list[i][0]
        else:
            # anchor number of multi levels
            num_level_anchors = [
                anchors.reshape([-1, self.box_code_size]).shape[0]
                for anchors in anchor_list[0]
            ]
            # concat all level anchors and flags to a single tensor
            for i in range(num_imgs):
                anchor_list[i] = paddle.concat(anchor_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         all_dir_targets, all_dir_weights, pos_inds_list,
         neg_inds_list) = multi_apply(
             self.anchor_target_3d_single,
             anchor_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             input_metas,
             label_channels=label_channels,
             num_classes=num_classes,
             sampling=sampling)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        dir_targets_list = images_to_levels(all_dir_targets, num_level_anchors)
        dir_weights_list = images_to_levels(all_dir_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, dir_targets_list, dir_weights_list,
                num_total_pos, num_total_neg)

    def anchor_target_3d_single(self,
                                anchors,
                                gt_bboxes,
                                gt_bboxes_ignore,
                                gt_labels,
                                input_meta,
                                label_channels=1,
                                num_classes=1,
                                sampling=True):
        """Compute targets of anchors in single batch.

        Args:
            anchors (paddle.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (paddle.Tensor): Ignored gt bboxes.
            gt_labels (paddle.Tensor): Gt class labels.
            input_meta (dict): Meta info of each image.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[paddle.Tensor]: Anchor targets.
        """
        if isinstance(self.bbox_assigner,
                      list) and (not isinstance(anchors, list)):
            feat_size = anchors.shape[0] * anchors.shape[1] * anchors.shape[2]
            rot_angles = anchors.shape[-2]
            assert len(self.bbox_assigner) == anchors.shape[-3]
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights,
             total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[..., i, :, :].reshape(
                    [-1, self.box_code_size])
                current_anchor_num += current_anchors.shape[0]
                if self.assign_per_class:
                    gt_per_cls = (gt_labels == i)
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes[gt_per_cls, :],
                        gt_bboxes_ignore, gt_labels[gt_per_cls], input_meta,
                        num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes, gt_bboxes_ignore,
                        gt_labels, input_meta, num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights, dir_targets,
                 dir_weights, pos_inds, neg_inds) = anchor_targets
                total_labels.append(labels.reshape([feat_size, 1, rot_angles]))
                total_label_weights.append(
                    label_weights.reshape([feat_size, 1, rot_angles]))
                total_bbox_targets.append(
                    bbox_targets.reshape(
                        [feat_size, 1, rot_angles, anchors.shape[-1]]))
                total_bbox_weights.append(
                    bbox_weights.reshape(
                        [feat_size, 1, rot_angles, anchors.shape[-1]]))
                total_dir_targets.append(
                    dir_targets.reshape([feat_size, 1, rot_angles]))
                total_dir_weights.append(
                    dir_weights.reshape([feat_size, 1, rot_angles]))
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = paddle.concat(total_labels, axis=-2).reshape([-1])
            total_label_weights = paddle.concat(
                total_label_weights, axis=-2).reshape([-1])
            total_bbox_targets = paddle.concat(
                total_bbox_targets, axis=-3).reshape([-1, anchors.shape[-1]])
            total_bbox_weights = paddle.concat(
                total_bbox_weights, axis=-3).reshape([-1, anchors.shape[-1]])
            total_dir_targets = paddle.concat(
                total_dir_targets, axis=-2).reshape([-1])
            total_dir_weights = paddle.concat(
                total_dir_weights, axis=-2).reshape([-1])
            total_pos_inds = paddle.concat(total_pos_inds, axis=0).reshape([-1])
            total_neg_inds = paddle.concat(total_neg_inds, axis=0).reshape([-1])
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_pos_inds, total_neg_inds)
        elif isinstance(self.bbox_assigner, list) and isinstance(anchors, list):
            # class-aware anchors with different feature map sizes
            assert len(self.bbox_assigner) == len(anchors), \
                'The number of bbox assigners and anchors should be the same.'
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights,
             total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[i]
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_labels == i)
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes[gt_per_cls, :],
                        gt_bboxes_ignore, gt_labels[gt_per_cls], input_meta,
                        num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes, gt_bboxes_ignore,
                        gt_labels, input_meta, num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights, dir_targets,
                 dir_weights, pos_inds, neg_inds) = anchor_targets
                total_labels.append(labels)
                total_label_weights.append(label_weights)
                total_bbox_targets.append(
                    bbox_targets.reshape([-1, anchors[i].shape[-1]]))
                total_bbox_weights.append(
                    bbox_weights.reshape([-1, anchors[i].shape[-1]]))
                total_dir_targets.append(dir_targets)
                total_dir_weights.append(dir_weights)
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = paddle.concat(total_labels, axis=0)
            total_label_weights = paddle.concat(total_label_weights, axis=0)
            total_bbox_targets = paddle.concat(total_bbox_targets, axis=0)
            total_bbox_weights = paddle.concat(total_bbox_weights, axis=0)
            total_dir_targets = paddle.concat(total_dir_targets, axis=0)
            total_dir_weights = paddle.concat(total_dir_weights, axis=0)
            total_pos_inds = paddle.concat(total_pos_inds, axis=0)
            total_neg_inds = paddle.concat(total_neg_inds, axis=0)
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_pos_inds, total_neg_inds)
        else:
            return self.anchor_target_single_assigner(
                self.bbox_assigner, anchors, gt_bboxes, gt_bboxes_ignore,
                gt_labels, input_meta, num_classes, sampling)

    def anchor_target_single_assigner(self,
                                      bbox_assigner,
                                      anchors,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      input_meta,
                                      num_classes=1,
                                      sampling=True):
        """Assign anchors and encode positive anchors.

        Args:
            bbox_assigner (BaseAssigner): assign positive and negative boxes.
            anchors (paddle.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (paddle.Tensor): Ignored gt bboxes.
            gt_labels (paddle.Tensor): Gt class labels.
            input_meta (dict): Meta info of each image.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[paddle.Tensor]: Anchor targets.
        """
        anchors = anchors.reshape([-1, anchors.shape[-1]])
        num_valid_anchors = anchors.shape[0]
        bbox_targets = paddle.zeros_like(anchors)
        bbox_weights = paddle.zeros_like(anchors)
        dir_targets = paddle.zeros([anchors.shape[0]], dtype='int64')
        dir_weights = paddle.zeros([anchors.shape[0]], dtype='float32')
        labels = paddle.zeros([num_valid_anchors], dtype='int64')
        label_weights = paddle.zeros([num_valid_anchors], dtype='float32')
        if len(gt_bboxes) > 0:
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
            sampling_result = self.bbox_sampler.sample(assign_result, anchors,
                                                       gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
        else:
            pos_inds = paddle.nonzero(
                paddle.zeros([
                    anchors.shape[0],
                ], dtype='bool') > 0,
                as_tuple=False).squeeze(-1).unique()
            neg_inds = paddle.nonzero(
                paddle.zeros([
                    anchors.shape[0],
                ], dtype='bool') == 0,
                as_tuple=False).squeeze(-1).unique()

        if gt_labels is not None:
            labels += num_classes
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            pos_dir_targets = get_direction_target(
                sampling_result.pos_bboxes,
                pos_bbox_targets,
                self.dir_offset,
                one_hot=False)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dir_targets[pos_inds] = pos_dir_targets
            dir_weights[pos_inds] = 1.0

            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']

        if len(neg_inds) > 0:
            # FIXME: label_weights[neg_inds] = 1.0 is too slow.
            label_weights_np = label_weights.numpy()
            label_weights_np[neg_inds.numpy()] = 1.0
            label_weights = paddle.to_tensor(label_weights_np)
            # label_weights = paddle.scatter(label_weights, neg_inds, paddle.ones())
        return (labels, label_weights, bbox_targets, bbox_weights, dir_targets,
                dir_weights, pos_inds, neg_inds)


def get_direction_target(anchors,
                         reg_targets,
                         dir_offset=0,
                         num_bins=2,
                         one_hot=True):
    """Encode direction to 0 ~ num_bins-1.

    Args:
        anchors (paddle.Tensor): Concatenated multi-level anchor.
        reg_targets (paddle.Tensor): Bbox regression targets.
        dir_offset (int): Direction offset.
        num_bins (int): Number of bins to divide 2*PI.
        one_hot (bool): Whether to encode as one hot.

    Returns:
        paddle.Tensor: Encoded direction targets.
    """
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = paddle.floor(
        offset_rot / (2 * np.pi / num_bins)).astype('int64')
    dir_cls_targets = paddle.clip(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        raise NotImplementedError
    return dir_cls_targets
