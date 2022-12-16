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

import itertools
import random
from typing import Tuple

import numpy as np
from PIL import Image
from skimage import transform as trans

from paddle3d.apis import manager
from paddle3d.geometries.bbox import BBoxes3D, second_box_encode
from paddle3d.sample import Sample
from paddle3d.transforms import functional as F
from paddle3d.transforms.base import TransformABC
"""
The smoke heatmap processing(encode_label/get_transfrom_matrix/affine_transform/get_3rd_point/gaussian_radius/gaussian2D/draw_umich_gaussian)
is based on https://github.com/lzccccc/SMOKE/blob/master/smoke/modeling/heatmap_coder.py
Ths copyright is MIT License
"""


def encode_label(K, ry, dims, locs):
    """get bbox 3d and 2d by model output

    Args:
        K (np.ndarray): camera intrisic matrix
        ry (np.ndarray): rotation y
        dims (np.ndarray): dimensions
        locs (np.ndarray): locations
    """
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += -np.float32(l) / 2
    y_corners += -np.float32(h)
    z_corners += -np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])

    loc_center = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, loc_center)
    proj_point = proj_point[:2] / proj_point[2]

    corners_2d = np.matmul(K, corners_3d)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array([
        min(corners_2d[0]),
        min(corners_2d[1]),
        max(corners_2d[0]),
        max(corners_2d[1])
    ])

    return proj_point, box2d, corners_3d


def get_transfrom_matrix(center_scale, output_size):
    """get transform matrix
    """
    center, scale = center_scale[0], center_scale[1]
    # todo: further add rot and shift here.
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    src_dir = np.array([0, src_w * -0.5])
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    get_matrix = trans.estimate_transform("affine", src, dst)
    matrix = get_matrix.params

    return matrix.astype(np.float32)


def affine_transform(point, matrix):
    """do affine transform to label
    """
    point_exd = np.array([point[0], point[1], 1.])
    new_point = np.matmul(matrix, point_exd)

    return new_point[:2]


def get_3rd_point(point_a, point_b):
    """get 3rd point
    """
    d = point_a - point_b
    point_c = point_b + np.array([-d[1], d[0]])
    return point_c


def gaussian_radius(h, w, thresh_min=0.7):
    """gaussian radius
    """
    a1 = 1
    b1 = h + w
    c1 = h * w * (1 - thresh_min) / (1 + thresh_min)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - thresh_min) * w * h
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * thresh_min
    b3 = -2 * thresh_min * (h + w)
    c3 = (thresh_min - 1) * w * h
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """get 2D gaussian map
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """draw umich gaussian
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius -
                               left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


@manager.TRANSFORMS.add_component
class Gt2SmokeTarget(TransformABC):
    def __init__(self,
                 mode: str,
                 num_classes: int,
                 flip_prob: float = 0.5,
                 aug_prob: float = 0.3,
                 max_objs: int = 50,
                 input_size: Tuple[int, int] = (1280, 384),
                 output_stride: Tuple[int, int] = (4, 4),
                 shift_range: Tuple[float, float, float] = (),
                 scale_range: Tuple[float, float, float] = ()):
        self.max_objs = max_objs
        self.input_width = input_size[0]
        self.input_height = input_size[1]
        self.output_width = self.input_width // output_stride[0]
        self.output_height = self.input_height // output_stride[1]

        self.shift_range = shift_range
        self.scale_range = scale_range
        self.shift_scale = (0.2, 0.4)
        self.flip_prob = flip_prob
        self.aug_prob = aug_prob
        self.is_train = True if mode == 'train' else False
        self.num_classes = num_classes

    def __call__(self, sample: Sample):
        img = Image.fromarray(sample.data)
        K = sample.meta.camera_intrinsic
        bboxes_3d = sample.bboxes_3d
        labels = sample.labels

        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        flipped = False
        if (self.is_train) and (random.random() < self.flip_prob):
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = size[0] - center[0] - 1
            K[0, 2] = size[0] - K[0, 2] - 1

        affine = False
        if (self.is_train) and (random.random() < self.aug_prob):
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)

        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size, [self.input_width, self.input_height])
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center_size, [self.output_width, self.output_height])

        if not self.is_train:
            # for inference we parametrize with original size
            target = {}
            target["image_size"] = size
            target["is_train"] = self.is_train
            target["trans_mat"] = trans_mat
            target["K"] = K
            sample.target = target
            sample.data = np.array(img)
            return sample

        heat_map = np.zeros(
            [self.num_classes, self.output_height, self.output_width],
            dtype=np.float32)
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
        p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        c_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        flip_mask = np.zeros([self.max_objs], dtype=np.uint8)
        bbox2d_size = np.zeros([self.max_objs, 2], dtype=np.float32)

        for i, (box3d, label) in enumerate(zip(bboxes_3d, labels)):
            if i == self.max_objs:
                break

            locs = np.array(box3d[0:3])
            rot_y = np.array(box3d[6])
            if flipped:
                locs[0] *= -1
                rot_y *= -1

            height, width, length = box3d[3:6]
            point, box2d, box3d = encode_label(
                K, rot_y, np.array([length, height, width]), locs)
            if np.all(box2d == 0):
                continue
            point = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]
            center = np.array([(box2d[0] + box2d[2]) / 2,
                               (box2d[1] + box2d[3]) / 2],
                              dtype=np.float32)

            if (0 < center[0] < self.output_width) and (0 < center[1] <
                                                        self.output_height):
                point_int = center.astype(np.int32)
                p_offset = point - point_int
                c_offset = center - point_int
                radius = gaussian_radius(h, w)
                radius = max(0, int(radius))
                heat_map[label] = draw_umich_gaussian(heat_map[label],
                                                      point_int, radius)

                cls_ids[i] = label
                regression[i] = box3d
                proj_points[i] = point_int
                p_offsets[i] = p_offset
                c_offsets[i] = c_offset
                dimensions[i] = np.array([length, height, width])
                locations[i] = locs
                rotys[i] = rot_y
                reg_mask[i] = 1 if not affine else 0
                flip_mask[i] = 1 if not affine and flipped else 0

                # targets for 2d bbox
                bbox2d_size[i, 0] = w
                bbox2d_size[i, 1] = h

        target = {}
        target["image_size"] = np.array(img.size)
        target["is_train"] = self.is_train
        target["trans_mat"] = trans_mat
        target["K"] = K
        target["hm"] = heat_map
        target["reg"] = regression
        target["cls_ids"] = cls_ids
        target["proj_p"] = proj_points
        target["dimensions"] = dimensions
        target["locations"] = locations
        target["rotys"] = rotys
        target["reg_mask"] = reg_mask
        target["flip_mask"] = flip_mask
        target["bbox_size"] = bbox2d_size
        target["c_offsets"] = c_offsets

        sample.target = target
        sample.data = np.array(img)
        return sample


@manager.TRANSFORMS.add_component
class Gt2CenterPointTarget(TransformABC):
    def __init__(self,
                 tasks: Tuple[dict],
                 down_ratio: int,
                 point_cloud_range: Tuple[float],
                 voxel_size: Tuple[float],
                 gaussian_overlap: float = 0.1,
                 max_objs: int = 500,
                 min_radius: int = 2):
        self.tasks = tasks
        self.down_ratio = down_ratio
        self.gaussian_overlap = gaussian_overlap
        self.max_objs = max_objs
        self.min_radius = min_radius
        self.voxel_size_x, self.voxel_size_y, self.voxel_size_z = voxel_size[0:
                                                                             3]
        self.point_cloud_range_x_min, self.point_cloud_range_y_min, self.point_cloud_range_z_min = point_cloud_range[
            0:3]
        self.point_cloud_range_x_max, self.point_cloud_range_y_max, self.point_cloud_range_z_max = point_cloud_range[
            3:6]
        self.grid_size_x = int(
            round((point_cloud_range[3] - point_cloud_range[0]) /
                  self.voxel_size_x))
        self.grid_size_y = int(
            round((point_cloud_range[4] - point_cloud_range[1]) /
                  self.voxel_size_y))
        self.grid_size_z = int(
            round((point_cloud_range[5] - point_cloud_range[2]) /
                  self.voxel_size_z))
        self.num_classes_by_task = [task["num_class"] for task in tasks]
        self.class_names_by_task = [task["class_names"] for task in tasks]
        self.all_class_names = list(itertools.chain(*self.class_names_by_task))

    def _gaussian_radius(self, height, width, min_overlap=0.5):
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def __call__(self, sample: Sample):
        # Get feature map size
        feature_map_size_x = self.grid_size_x // self.down_ratio
        feature_map_size_y = self.grid_size_y // self.down_ratio

        # Reorder the bboxes_3d and labels for each task
        labels = sample.labels
        bboxes_3d = sample.bboxes_3d
        velocities = getattr(sample.bboxes_3d, "velocities", None)
        bboxes_3d_origin = sample.bboxes_3d.origin
        required_origin = [0.5, 0.5, 0.5]
        if list(bboxes_3d_origin) != required_origin:
            bboxes_3d_origin = np.asarray(bboxes_3d_origin)
            required_origin = np.asarray([0.5, 0.5, 0.5])
            bboxes_3d[..., :3] += bboxes_3d[..., 3:6] * (
                required_origin - bboxes_3d_origin)

        bboxes_3d_by_task = []
        labels_by_task = []
        velocities_by_task = []

        task_label_begin = 0
        for task_idx, class_names in enumerate(self.class_names_by_task):
            task_bboxes_3d = []
            task_labels = []
            task_velocities = []
            for class_name in class_names:
                mask = np.where(
                    labels == self.all_class_names.index(class_name))
                task_bboxes_3d.append(bboxes_3d[mask])
                task_labels.append(labels[mask] - task_label_begin)
                if velocities is not None:
                    task_velocities.append(velocities[mask])
            task_label_begin += len(class_names)
            bboxes_3d_by_task.append(np.concatenate(task_bboxes_3d, axis=0))
            labels_by_task.append(np.concatenate(task_labels))
            if velocities is not None:
                velocities_by_task.append(
                    np.concatenate(task_velocities, axis=0))

        # Limit the orientation angle within [-np.pi, +np.pi]
        for task_bboxes_3d in bboxes_3d_by_task:
            task_bboxes_3d[:, -1] = BBoxes3D.limit_period(
                task_bboxes_3d[:, -1], offset=0.5, period=np.pi * 2)

        heat_maps, target_bboxs, center_idxs, target_masks, target_labels = [], [], [], [], []
        for task_idx, task in enumerate(self.tasks):
            heat_map = np.zeros((len(self.class_names_by_task[task_idx]),
                                 feature_map_size_y, feature_map_size_x),
                                dtype=np.float32)

            # [x, y, z, w, l, h, vx, vy, rots, rotc]
            if len(velocities_by_task) > 0:
                target_bbox = np.zeros((self.max_objs, 10), dtype=np.float32)
            else:
                target_bbox = np.zeros((self.max_objs, 8), dtype=np.float32)
            center_idx = np.zeros((self.max_objs), dtype=np.int64)
            target_mask = np.zeros((self.max_objs), dtype=np.uint8)
            target_label = np.zeros((self.max_objs), dtype=np.int64)

            num_objs = min(bboxes_3d_by_task[task_idx].shape[0], self.max_objs)

            for obj_idx in range(num_objs):
                cls_id = labels_by_task[task_idx][obj_idx]

                w, l, h = bboxes_3d_by_task[task_idx][obj_idx][3:6]
                w = w / self.voxel_size_x / self.down_ratio
                l = l / self.voxel_size_y / self.down_ratio
                if w > 0 and l > 0:
                    radius = self._gaussian_radius(
                        l, w, min_overlap=self.gaussian_overlap)
                    radius = max(self.min_radius, int(radius))

                    x, y, z = bboxes_3d_by_task[task_idx][obj_idx][0:3]
                    center = np.array([(x - self.point_cloud_range_x_min) /
                                       self.voxel_size_x / self.down_ratio,
                                       (y - self.point_cloud_range_y_min) /
                                       self.voxel_size_y / self.down_ratio],
                                      dtype=np.float32)
                    center_int = center.astype(np.int32)

                    if not (0 <= center_int[0] < feature_map_size_x
                            and 0 <= center_int[1] < feature_map_size_y):
                        continue

                    draw_umich_gaussian(heat_map[cls_id], center, radius)
                    target_label[obj_idx] = cls_id
                    center_idx[obj_idx] = center_int[
                        1] * feature_map_size_x + center_int[0]
                    target_mask[obj_idx] = 1

                    angle = bboxes_3d_by_task[task_idx][obj_idx][-1]
                    if len(velocities_by_task) > 0:
                        vx, vy = velocities_by_task[task_idx][obj_idx][0:2]
                        target_bbox[obj_idx] = np.concatenate(
                            (center - center_int, z,
                             np.log(bboxes_3d_by_task[task_idx][obj_idx][3:6]),
                             np.array(vx), np.array(vy), np.sin(angle),
                             np.cos(angle)),
                            axis=None)
                    else:
                        target_bbox[obj_idx] = np.concatenate(
                            (center - center_int, z,
                             np.log(bboxes_3d_by_task[task_idx][obj_idx][3:6]),
                             np.sin(angle), np.cos(angle)),
                            axis=None)

            heat_maps.append(heat_map)
            target_bboxs.append(target_bbox)
            target_masks.append(target_mask)
            center_idxs.append(center_idx)
            target_labels.append(target_label)

        sample.heat_map = heat_maps
        sample.target_bbox = target_bboxs
        sample.center_idx = center_idxs
        sample.target_mask = target_masks
        sample.target_label = target_labels

        sample.pop('bboxes_2d', None)
        sample.pop('bboxes_3d', None)
        sample.pop('path', None)
        sample.pop('labels', None)
        sample.pop('attrs', None)
        sample.pop('ignored_bboxes_3d', None)
        return sample


@manager.TRANSFORMS.add_component
class Gt2PointPillarsTarget(object):
    """
    Assign ground truth to anchors.

    Args:
        positive_fraction (float, optional): None or a float between 0 and 1. If not None, the ratio between the
            number of positive samples and the number of negative samples will be kept to `positive_fraction`.
            If there are not enough positives, fill the rest with negatives.
        rpn_batch_size (int, optional): Sample size. Defaults to 512.
        norm_by_num_examples (bool, optional): Whether to normalize box_weight by number of samples.
            Defaults to False.
    """

    def __init__(self,
                 positive_fraction=None,
                 rpn_batch_size=512,
                 norm_by_num_examples=False):
        self.region_similarity_calculator = F.nearest_iou_similarity
        self.positive_fraction = positive_fraction
        self.rpn_batch_size = rpn_batch_size
        self.norm_by_num_examples = norm_by_num_examples

    def assign(self,
               all_anchors,
               gt_boxes,
               anchors_mask=None,
               gt_classes=None,
               matched_thresholds=0.6,
               unmatched_thresholds=0.45):
        """
        Calculate the target for each sample.

        Args:
            all_anchors: [num_of_anchors, box_ndim] float array.
            gt_boxes: [num_gt_boxes, box_ndim] float array.
            anchors_mask: Bool array indicates valid anchors.
            gt_classes: [num_gt_boxes] int array. indicate gt classes, must
                start with 1.
            matched_thresholds: float, iou less than matched_threshold will
                be treated as positives.
            unmatched_thresholds: float, iou less than unmatched_threshold will
                be treated as negatives.

        Returns:
            labels, reg_targets, reg_weights
        """

        total_anchors = all_anchors.shape[0]
        if anchors_mask is not None:
            # Filter out invalid anchors whose area < threshold
            inds_inside = np.where(anchors_mask)[0]
            anchors = all_anchors[inds_inside, :]
            if not isinstance(matched_thresholds, float):
                matched_thresholds = matched_thresholds[inds_inside]
            if not isinstance(unmatched_thresholds, float):
                unmatched_thresholds = unmatched_thresholds[inds_inside]
        else:
            anchors = all_anchors

        num_valid_anchors = len(anchors)

        if gt_classes is None:
            gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)
        # Compute anchor labels:
        # 0 is negative, -1 is don't care (ignore)
        labels = np.full((num_valid_anchors, ), -1, dtype=np.int32)
        gt_ids = np.full((num_valid_anchors, ), -1, dtype=np.int32)

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # Compute overlaps between the anchors and the gt boxes overlaps
            anchor_by_gt_overlap = self.region_similarity_calculator(
                anchors, gt_boxes)
            # Map from anchor to gt box that has the highest overlap
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
            # For each anchor, amount of overlap with most overlapping gt box
            anchor_to_gt_max = anchor_by_gt_overlap[np.arange(
                num_valid_anchors), anchor_to_gt_argmax]
            # Map from gt box to an anchor that has the highest overlap
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
            # For each gt box, amount of overlap with most overlapping anchor
            gt_to_anchor_max = anchor_by_gt_overlap[
                gt_to_anchor_argmax,
                np.arange(anchor_by_gt_overlap.shape[1])]
            # must remove gt which doesn't match any anchor.
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1
            # Find all anchors that share the max overlap amount
            # (this includes many ties)
            anchors_with_max_overlap = np.where(
                anchor_by_gt_overlap == gt_to_anchor_max)[0]
            # Fg label: for each gt use anchors with the highest overlap
            # (including ties)
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force
            # Fg label: above threshold IOU
            pos_inds = anchor_to_gt_max >= matched_thresholds
            gt_inds = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds]
            gt_ids[pos_inds] = gt_inds
            bg_inds = np.where(anchor_to_gt_max < unmatched_thresholds)[0]
        else:
            bg_inds = np.arange(num_valid_anchors)

        fg_inds = np.where(labels > 0)[0]
        fg_max_overlap = None
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_max_overlap = anchor_to_gt_max[fg_inds]

        gt_pos_ids = gt_ids[fg_inds]

        # subsample positive labels if there are too many
        if self.positive_fraction is not None:
            num_fg = int(self.positive_fraction * self.rpn_batch_size)
            if len(fg_inds) > num_fg:
                disable_inds = np.random.choice(
                    fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels[disable_inds] = -1
                fg_inds = np.where(labels > 0)[0]

            # subsample negative labels if there are too many
            # (samples with replacement, but since the set of bg inds is large,
            # most samples will not have repeats)
            num_bg = self.rpn_batch_size - np.sum(labels > 0)
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[np.random.randint(
                    len(bg_inds), size=num_bg)]
                labels[enable_inds] = 0
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0
                # re-enable anchors_with_max_overlap
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        reg_targets = np.zeros((num_valid_anchors, all_anchors.shape[-1]),
                               dtype=all_anchors.dtype)

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            reg_targets[fg_inds, :] = second_box_encode(
                gt_boxes[anchor_to_gt_argmax[fg_inds], :], anchors[fg_inds, :])

        reg_weights = np.zeros((num_valid_anchors, ), dtype=all_anchors.dtype)
        # uniform weighting of examples (given non-uniform sampling)
        if self.norm_by_num_examples:
            num_examples = np.sum(labels >= 0)  # neg + pos
            num_examples = np.maximum(1.0, num_examples)
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        # Map up to original set of anchors
        if anchors_mask is not None:
            labels = self._unmap(labels, total_anchors, inds_inside, fill=-1)
            reg_targets = self._unmap(
                reg_targets, total_anchors, inds_inside, fill=0)
            reg_weights = self._unmap(
                reg_weights, total_anchors, inds_inside, fill=0)
        ret = {
            "labels": labels,
            "reg_targets": reg_targets,
            "reg_weights": reg_weights,
            "assigned_anchors_overlap": fg_max_overlap,
            "positive_gt_id": gt_pos_ids,
        }

        return ret

    def _unmap(self, data, count, inds, fill=0):
        """
        Unmap a subset of item (data) back to the original set of items (of size count)
        """
        if count == len(inds):
            return data

        if len(data.shape) == 1:
            ret = np.full((count, ), fill, dtype=data.dtype)
            ret[inds] = data
        else:
            ret = np.full((count, ) + data.shape[1:], fill, dtype=data.dtype)
            ret[inds, :] = data
        return ret

    def __call__(self, sample: Sample):
        sample.bboxes_3d[:, -1] = BBoxes3D.limit_period(
            sample.bboxes_3d[:, -1], offset=0.5, period=np.pi * 2)
        ret = self.assign(
            sample.anchors,
            sample.bboxes_3d,
            anchors_mask=sample.get("anchors_mask", None),
            gt_classes=sample.labels +
            1,  # background is regarded as class 0, thus shift labels
            matched_thresholds=sample.matched_thresholds,
            unmatched_thresholds=sample.unmatched_thresholds)

        sample.reg_targets = ret["reg_targets"]
        sample.reg_weights = ret["reg_weights"]

        sample.labels = ret["labels"]

        # the followings are not used in training
        sample.pop("anchors", None)
        sample.pop("bboxes_3d", None)
        sample.pop('path', None)
        sample.pop("difficulties", None)
        sample.pop("ignored_bboxes_3d", None)

        return sample


@manager.TRANSFORMS.add_component
class Gt2PVRCNNTarget(TransformABC):
    def __init__(self):
        pass

    def __call__(self, sample: Sample):
        # Reorder the bboxes_3d and labels for each task
        labels = sample.labels
        bboxes_3d = sample.bboxes_3d
        bboxes_3d_origin = sample.bboxes_3d.origin
        required_origin = [0.5, 0.5, 0.5]
        if list(bboxes_3d_origin) != required_origin:
            bboxes_3d_origin = np.asarray(bboxes_3d_origin)
            required_origin = np.asarray([0.5, 0.5, 0.5])
            bboxes_3d[..., :3] += bboxes_3d[..., 3:6] * (
                required_origin - bboxes_3d_origin)

        bboxes_3d[..., 3:5] = bboxes_3d[..., [4, 3]]
        bboxes_3d[..., -1] = -(bboxes_3d[..., -1] + np.pi / 2.)
        bboxes_3d[..., -1] = BBoxes3D.limit_period(
            bboxes_3d[..., -1], offset=0.5, period=2 * np.pi)
        labels = labels + 1
        gt_boxes = np.concatenate(
            (bboxes_3d, labels.reshape(-1, 1).astype(np.float32)), axis=1)
        sample.gt_boxes = gt_boxes

        sample.pop('bboxes_2d', None)
        sample.pop('bboxes_3d', None)
        sample.pop('path', None)
        sample.pop('labels', None)
        sample.pop('attrs', None)
        sample.pop('ignored_bboxes_3d', None)
        return sample
