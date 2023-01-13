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

import numpy as np
import paddle

from paddle3d.models.heads.dense_heads.anchor_mixins import limit_period
from paddle3d.sample import _EasyDict

__all__ = ['MaxIoUAssigner']


class BboxOverlapsNearest3D(object):
    """Nearest 3D IoU Calculator.

    Note:
        This IoU calculator first finds the nearest 2D boxes in bird eye view
        (BEV), and then calculates the 2D IoU using :meth:`bbox_overlaps`.

    Args:
        coordinate (str): 'camera', 'lidar', or 'depth' coordinate system.
    """

    def __init__(self, coordinate='lidar'):
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate nearest 3D IoU.

        Note:
            If ``is_aligned`` is ``False``, then it calculates the ious between
            each bbox of bboxes1 and bboxes2, otherwise it calculates the ious
            between each aligned pair of bboxes1 and bboxes2.

        Args:
            bboxes1 (paddle.Tensor): shape (N, 7+N) [x, y, z, h, w, l, ry, v].
            bboxes2 (paddle.Tensor): shape (M, 7+N) [x, y, z, h, w, l, ry, v].
            mode (str): "iou" (intersection over union) or iof
                (intersection over foreground).
            is_aligned (bool): Whether the calculation is aligned.

        Return:
            paddle.Tensor: If ``is_aligned`` is ``True``, return ious between \
                bboxes1 and bboxes2 with shape (M, N). If ``is_aligned`` is \
                ``False``, return shape is M.
        """
        return bbox_overlaps_nearest_3d(bboxes1, bboxes2, mode, is_aligned,
                                        self.coordinate)

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str


def nearest_bev(bboxes):
    bev_rotated_boxes = paddle.gather(
        bboxes, paddle.to_tensor([0, 1, 3, 4, 6]), axis=1)
    # convert the rotation to a valid range
    rotations = bev_rotated_boxes[:, -1]
    normed_rotations = paddle.abs(limit_period(rotations, 0.5, np.pi))

    # find the center of boxes
    conditions = (normed_rotations > np.pi / 4)[..., None]
    bboxes_xywh = paddle.where(
        conditions,
        paddle.gather(
            bev_rotated_boxes, paddle.to_tensor([0, 1, 3, 2]), axis=1),
        bev_rotated_boxes[:, :4])

    centers = bboxes_xywh[:, :2]
    dims = bboxes_xywh[:, 2:]
    bev_boxes = paddle.concat([centers - dims / 2, centers + dims / 2], axis=-1)
    return bev_boxes


def bbox_overlaps_nearest_3d(bboxes1,
                             bboxes2,
                             mode='iou',
                             is_aligned=False,
                             coordinate='lidar'):
    """Calculate nearest 3D IoU.
    """
    assert bboxes1.shape[-1] == bboxes2.shape[-1] >= 7

    # # Change the bboxes to bev
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)

    ret = bbox_overlaps(
        bboxes1_bev, bboxes2_bev, mode=mode, is_aligned=is_aligned)
    return ret


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        raise ValueError("dim is 0")

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = paddle.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = paddle.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = paddle.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = paddle.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = paddle.maximum(bboxes1[..., :, None, :2],
                            bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = paddle.minimum(bboxes1[..., :, None, 2:],
                            bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clip(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = paddle.minimum(bboxes1[..., :, None, :2],
                                         bboxes2[..., None, :, :2])
            enclosed_rb = paddle.maximum(bboxes1[..., :, None, 2:],
                                         bboxes2[..., None, :, 2:])

    eps = paddle.to_tensor([eps])
    union = paddle.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clip(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = paddle.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


class MaxIoUAssigner(object):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = BboxOverlapsNearest3D()

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps = ignore_overlaps.max(axis=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps = ignore_overlaps.max(axis=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.shape[0], overlaps.shape[1]

        # 1. assign -1 by default
        assigned_gt_inds = paddle.full([
            num_bboxes,
        ], -1, dtype='int64')

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = paddle.zeros([
                num_bboxes,
            ])
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = paddle.full([
                    num_bboxes,
                ], -1, dtype='int64')
            return _EasyDict(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps = overlaps.max(axis=0)
        argmax_overlaps = overlaps.argmax(axis=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps = overlaps.max(axis=1)
        gt_argmax_overlaps = overlaps.argmax(axis=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            # Low-quality matching will overwrite the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox B.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = paddle.full([
                num_bboxes,
            ], -1, dtype='int64')
            pos_inds = paddle.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                # zero shape tensor
                if pos_inds.ndim == 0:
                    pos_inds = pos_inds.unsqueeze(-1)
                try:
                    assigned_labels[pos_inds] = gt_labels[
                        assigned_gt_inds[pos_inds] - 1]
                except:
                    assigned_labels[pos_inds] = gt_labels[
                        assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return _EasyDict(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels)
