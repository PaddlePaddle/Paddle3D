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
from paddle3d.apis import manager

INF = 100000000.


@manager.MODELS.add_component
class DD3DTargetPreparer(nn.Layer):
    """
    This code is based on https://github.com/TRI-ML/dd3d/blob/main/tridet/modeling/dd3d/prepare_targets.py#L11
    """

    def __init__(self,
                 input_strides,
                 num_classes=5,
                 center_sample=True,
                 radius=1.5,
                 dd3d_on=True,
                 sizes_of_interest=[64, 128, 256, 512]):
        super(DD3DTargetPreparer, self).__init__()
        self.num_classes = num_classes
        self.center_sample = center_sample
        self.strides = input_strides
        self.radius = radius
        self.dd3d_enabled = dd3d_on

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in sizes_of_interest:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

    def forward(self, locations, bboxes_2d, bboxes_3d, labels, feature_shapes):
        # gt_instances
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = paddle.to_tensor(
                self.sizes_of_interest[l], dtype=loc_per_level.dtype)
            loc_to_size_range.append(loc_to_size_range_per_level[None].expand(
                [num_loc_list[l], -1]))

        loc_to_size_range = paddle.concat(loc_to_size_range, axis=0)
        locations = paddle.concat(locations, axis=0)

        training_targets = self.compute_targets_for_locations(
            locations, bboxes_2d, bboxes_3d, labels, loc_to_size_range,
            num_loc_list)

        training_targets["locations"] = [
            locations.clone() for _ in range(bboxes_2d.shape[0])
        ]
        training_targets["im_inds"] = [
            paddle.ones([locations.shape[0]], dtype='int64') * i
            for i in range(bboxes_2d.shape[0])
        ]

        box2d = training_targets.pop("box2d", None)

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(k, v, num_loc_list)
            for k, v in training_targets.items() if k != "box2d"
        }

        training_targets["fpn_levels"] = [
            paddle.ones([len(loc)], dtype='int64') * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # Flatten targets: (L x B x H x W, TARGET_SIZE)
        labels = paddle.concat(
            [x.reshape([-1]) for x in training_targets["labels"]], axis=0)
        box2d_reg_targets = paddle.concat(
            [x.reshape([-1, 4]) for x in training_targets["box2d_reg"]], axis=0)

        target_inds = paddle.concat(
            [x.reshape([-1]) for x in training_targets["target_inds"]], axis=0)
        locations = paddle.concat(
            [x.reshape([-1, 2]) for x in training_targets["locations"]], axis=0)
        im_inds = paddle.concat(
            [x.reshape([-1]) for x in training_targets["im_inds"]], axis=0)
        fpn_levels = paddle.concat(
            [x.reshape([-1]) for x in training_targets["fpn_levels"]], axis=0)

        pos_inds = paddle.nonzero(labels != self.num_classes).squeeze(1)

        targets = {
            "labels": labels,
            "box2d_reg_targets": box2d_reg_targets,
            "locations": locations,
            "target_inds": target_inds,
            "im_inds": im_inds,
            "fpn_levels": fpn_levels,
            "pos_inds": pos_inds
        }

        if self.dd3d_enabled:
            box3d_targets = paddle.concat(
                [x.reshape([-1, 10]) for x in training_targets["box3d"]],
                axis=0)
            # box3d_targets = Boxes3D.cat(training_targets["box3d"])
            targets.update({"box3d_targets": box3d_targets})

            if box2d is not None:
                # Original format is B x L x (H x W, 4)
                # Need to be in L x (B, 4, H, W).
                batched_box2d = []
                for lvl, per_lvl_box2d in enumerate(zip(*box2d)):
                    # B x (H x W, 4)
                    h, w = feature_shapes[lvl]
                    batched_box2d_lvl = paddle.stack(
                        [x.T.reshape([4, h, w]) for x in per_lvl_box2d], axis=0)
                    batched_box2d.append(batched_box2d_lvl)
                targets.update({"batched_box2d": batched_box2d})

        return targets

    def compute_targets_for_locations(self, locations, bboxes_2d, bboxes_3d,
                                      labels_batch, size_ranges, num_loc_list):
        # targets
        labels = []
        box2d_reg = []

        if self.dd3d_enabled:
            box3d = []

        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(bboxes_2d.shape[0]):
            bboxes = bboxes_2d[im_i, ...]
            labels_per_im = labels_batch[im_i, ...]

            # no gt
            if bboxes.numel() == 0:
                labels.append(
                    paddle.zeros([locations.shape[0]]) + self.num_classes)
                # reg_targets.append(paddle.zeros((locations.shape[0], 4)))
                box2d_reg.append(paddle.zeros((locations.shape[0], 4)))
                target_inds.append(paddle.zeros([locations.shape[0]]) - 1)

                if self.dd3d_enabled:
                    box3d.append(paddle.zeros((locations.shape[0], 10)))
                continue

            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            box2d_reg_per_im = paddle.stack([l, t, r, b], axis=2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(bboxes, num_loc_list, xs,
                                                     ys)
            else:
                is_in_boxes = box2d_reg_per_im.min(axis=2) > 0

            max_reg_targets_per_im = box2d_reg_per_im.max(axis=2)
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, 0:1]) & \
                (max_reg_targets_per_im <= size_ranges[:, 1:2])
            locations_to_gt_area = area[None].tile([len(locations), 1])

            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area = locations_to_gt_area.min(axis=1)
            locations_to_gt_inds = locations_to_gt_area.argmin(axis=1)

            indes = paddle.stack(
                [paddle.arange(len(locations)), locations_to_gt_inds], 1)
            box2d_reg_per_im = paddle.gather_nd(box2d_reg_per_im, indes)
            # box2d_reg_per_im = box2d_reg_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += bboxes_2d.shape[0]

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            box2d_reg.append(box2d_reg_per_im)
            target_inds.append(target_inds_per_im)

            if self.dd3d_enabled:
                # 3D box targets
                box3d_per_im = bboxes_3d[im_i, ...][locations_to_gt_inds]
                box3d.append(box3d_per_im)

        ret = {
            "labels": labels,
            "box2d_reg": box2d_reg,
            "target_inds": target_inds
        }
        if self.dd3d_enabled:
            ret.update({"box3d": box3d})

        return ret

    def get_sample_region(self, boxes, num_loc_list, loc_xs, loc_ys):
        center_x = boxes[:, 0::2].sum(axis=-1) * 0.5
        center_y = boxes[:, 1::2].sum(axis=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand([K, num_gts, 4])
        center_x = center_x[None].expand([K, num_gts])
        center_y = center_y[None].expand([K, num_gts])
        center_gt = paddle.zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return paddle.zeros(loc_xs.shape).cast('bool')
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = self.strides[level] * self.radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = paddle.where(xmin > boxes[beg:end, :, 0],
                                                    xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = paddle.where(ymin > boxes[beg:end, :, 1],
                                                    ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = paddle.where(xmax > boxes[beg:end, :, 2],
                                                    boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = paddle.where(ymax > boxes[beg:end, :, 3],
                                                    boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = paddle.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1) > 0
        return inside_gt_bbox_mask

    def _transpose(self, k, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        if k == "box3d":
            for im_i in range(len(training_targets)):
                # training_targets[im_i] = paddle.split(training_targets[im_i], num_loc_list, axis=0)
                training_targets[im_i] = training_targets[im_i].split(
                    num_loc_list, axis=0)

            targets_level_first = []
            for targets_per_level in zip(*training_targets):
                targets_level_first.append(
                    paddle.concat(targets_per_level, axis=0))
            return targets_level_first

        for im_i in range(len(training_targets)):
            training_targets[im_i] = paddle.split(
                training_targets[im_i], num_loc_list, axis=0)

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(paddle.concat(targets_per_level, axis=0))
        return targets_level_first
