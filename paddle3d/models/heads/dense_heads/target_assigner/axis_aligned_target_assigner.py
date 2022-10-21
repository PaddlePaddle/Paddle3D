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

import numpy as np
import paddle

from paddle3d.utils.box import boxes3d_nearest_bev_iou


class AxisAlignedTargetAssigner(object):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/models/dense_heads/target_assigner/axis_aligned_target_assigner.py#L8
    """

    def __init__(self, anchor_generator_cfg, anchor_target_cfg, class_names,
                 box_coder):
        super().__init__()

        self.anchor_generator_cfg = anchor_generator_cfg
        self.anchor_target_cfg = anchor_target_cfg
        self.box_coder = box_coder
        self.class_names = np.array(class_names)
        self.anchor_class_names = [
            config['class_name'] for config in anchor_generator_cfg
        ]
        self.pos_fraction = anchor_target_cfg[
            'pos_fraction'] if anchor_target_cfg['pos_fraction'] >= 0 else None
        self.sample_size = anchor_target_cfg['sample_size']
        self.norm_by_num_examples = anchor_target_cfg['norm_by_num_examples']
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[
                config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[
                config['class_name']] = config['unmatched_threshold']

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """

        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].cast("int32")
            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names,
                                                  all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    mask = paddle.to_tensor(
                        self.class_names[cur_gt_classes.cpu().numpy() -
                                         1] == anchor_class_name)
                else:
                    mask = paddle.to_tensor([
                        self.class_names[c - 1] == anchor_class_name
                        for c in cur_gt_classes
                    ],
                                            dtype=paddle.bool)

                feature_map_size = anchors.shape[:3]
                anchors = anchors.reshape([-1, anchors.shape[-1]])
                selected_classes = cur_gt_classes[mask]

                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.
                    matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.
                    unmatched_thresholds[anchor_class_name])

                target_list.append(single_target)

            target_dict = {
                'box_cls_labels': [
                    t['box_cls_labels'].reshape([*feature_map_size, -1])
                    for t in target_list
                ],
                'box_reg_targets': [
                    t['box_reg_targets'].reshape(
                        [*feature_map_size, -1, self.box_coder.code_size])
                    for t in target_list
                ],
                'reg_weights': [
                    t['reg_weights'].reshape([*feature_map_size, -1])
                    for t in target_list
                ]
            }
            target_dict['box_reg_targets'] = paddle.concat(
                target_dict['box_reg_targets'],
                axis=-2).reshape([-1, self.box_coder.code_size])

            target_dict['box_cls_labels'] = paddle.concat(
                target_dict['box_cls_labels'], axis=-1).reshape([-1])
            target_dict['reg_weights'] = paddle.concat(
                target_dict['reg_weights'], axis=-1).reshape([-1])

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        bbox_targets = paddle.stack(bbox_targets, axis=0)

        cls_labels = paddle.stack(cls_labels, axis=0)
        reg_weights = paddle.stack(reg_weights, axis=0)
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights
        }
        return all_targets_dict

    def assign_targets_single(self,
                              anchors,
                              gt_boxes,
                              gt_classes,
                              matched_threshold=0.6,
                              unmatched_threshold=0.45):

        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = paddle.ones((num_anchors, ), dtype="int32") * -1
        gt_ids = paddle.ones((num_anchors, ), dtype="int32") * -1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = boxes3d_nearest_bev_iou(
                anchors[:, 0:7], gt_boxes[:, 0:7])

            anchor_to_gt_argmax = paddle.to_tensor(
                anchor_by_gt_overlap.cpu().numpy().argmax(axis=1))
            anchor_to_gt_max = anchor_by_gt_overlap[paddle.arange(num_anchors),
                                                    anchor_to_gt_argmax]

            gt_to_anchor_argmax = paddle.to_tensor(
                anchor_by_gt_overlap.cpu().numpy().argmax(axis=0))
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax,
                                                    paddle.arange(num_gt)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (
                anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            if anchors_with_max_overlap.shape[0] > 0:
                gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
                gt_ids[anchors_with_max_overlap] = gt_inds_force.cast("int32")

            pos_inds = paddle.where(anchor_to_gt_max >= matched_threshold)[0]

            if pos_inds.shape[0] > 0:
                gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
                labels[pos_inds] = gt_classes[gt_inds_over_thresh]
                gt_ids[pos_inds] = gt_inds_over_thresh.cast("int32")
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = paddle.arange(num_anchors)

        fg_inds = (labels > 0).nonzero()
        if self.pos_fraction is not None and fg_inds.numel() > 0:
            # TODO(qianhui): zero shape
            fg_inds = fg_inds[:, 0]
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = paddle.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[paddle.randint(
                    0, len(bg_inds), size=(num_bg, ))]
                labels[enable_inds] = 0
            # bg_inds = paddle.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                if fg_inds.numel() > 0:
                    fg_inds = fg_inds[:, 0]
                bg_inds = bg_inds.cast('int32')
                updates = paddle.zeros(bg_inds.shape, dtype='int32')
                labels = paddle.scatter(
                    labels.astype('int32'), index=bg_inds, updates=updates)
                # labels[bg_inds] = 0
                # labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        bbox_targets = paddle.zeros(
            shape=[num_anchors, self.box_coder.code_size], dtype=anchors.dtype)
        if gt_boxes.shape[0] > 0 and anchors.shape[0] > 0 and len(fg_inds) > 0:
            fg_gt_boxes = paddle.gather(
                gt_boxes, index=anchor_to_gt_argmax[fg_inds], axis=0)
            fg_anchors = paddle.gather(anchors, index=fg_inds, axis=0)
            bbox_targets[fg_inds, :] = self.box_coder.encode_paddle(
                fg_gt_boxes, fg_anchors)

        reg_weights = paddle.zeros(shape=[num_anchors], dtype=anchors.dtype)

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict
