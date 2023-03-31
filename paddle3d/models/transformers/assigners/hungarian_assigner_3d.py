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

# ------------------------------------------------------------------------
# This code is based on https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_3d.py
# Ths copyright of fundamentalvision/BEVFormer is as follows:
# Apache-2.0 license [see LICENSE for details].
# ------------------------------------------------------------------------

import numpy as np
import paddle
from scipy.optimize import linear_sum_assignment

from paddle3d.apis import manager
from paddle3d.utils.box import normalize_bbox


def ensure_rng(rng=None):
    """Coerces input into a random number generator.

    If the input is None, then a global random state is returned.

    If the input is a numeric value, then that is used as a seed to construct a
    random state. Otherwise the input is returned as-is.

    Adapted from [1]_.

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class
    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator

    References:
        .. [1] https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270  # noqa: E501
    """

    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng


class AssignResult:
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = paddle.LongTensor([0, .5, .9, 0])
        >>> gt_inds = paddle.LongTensor([-1, 1, 2, 0])
        >>> labels = paddle.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = paddle.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f'num_gts={self.num_gts!r}')
        if self.gt_inds is None:
            parts.append(f'gt_inds={self.gt_inds!r}')
        else:
            parts.append(f'gt_inds.shape={tuple(self.gt_inds.shape)!r}')
        if self.max_overlaps is None:
            parts.append(f'max_overlaps={self.max_overlaps!r}')
        else:
            parts.append('max_overlaps.shape='
                         f'{tuple(self.max_overlaps.shape)!r}')
        if self.labels is None:
            parts.append(f'labels={self.labels!r}')
        else:
            parts.append(f'labels.shape={tuple(self.labels.shape)!r}')
        return ', '.join(parts)

    @classmethod
    def random(cls, **kwargs):
        """Create random AssignResult for tests or debugging.

        Args:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assigned to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state

        Returns:
            :obj:`AssignResult`: Randomly generated assign results.

        Example:
            >>> from mmdet.core.bbox.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        rng = ensure_rng(kwargs.get('rng', None))

        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        p_use_label = kwargs.get('p_use_label', 0.5)
        num_classes = kwargs.get('p_use_label', 3)

        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            max_overlaps = paddle.zeros([num_preds], dtype=paddle.float32)
            gt_inds = paddle.zeros([num_preds], dtype=paddle.int64)
            if p_use_label is True or p_use_label < rng.rand():
                labels = paddle.zeros([num_preds], dtype=paddle.int64)
            else:
                labels = None
        else:
            import numpy as np

            # Create an overlap for each predicted box
            max_overlaps = paddle.to_tensor(rng.rand(num_preds))

            # Construct gt_inds for each predicted box
            is_assigned = rng.rand(num_preds) < p_assigned
            # maximum number of assignments constraints
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))

            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned = paddle.to_tensor(is_assigned)
            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True

            is_ignore = paddle.to_tensor(
                rng.rand(num_preds) < p_ignore) & is_assigned

            gt_inds = paddle.zeros([num_preds], dtype=paddle.int64)

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = paddle.to_tensor(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned]

            gt_inds = paddle.to_tensor(
                rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = paddle.zeros([num_preds], dtype=paddle.int64)
                else:
                    labels = paddle.to_tensor(
                        # remind that we set FG labels to [0, num_class-1]
                        # since mmdet v2.0
                        # BG cat_id: num_class
                        rng.randint(0, num_classes, size=num_preds))
                    labels[~is_assigned] = 0
            else:
                labels = None

        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.

        Args:
            gt_labels (paddle.Tensor): Labels of gt boxes
        """
        self_inds = paddle.arange(1, len(gt_labels) + 1, dtype=paddle.int64)
        self.gt_inds = paddle.concat([self_inds, self.gt_inds])

        self.max_overlaps = paddle.concat(
            [paddle.ones((len(gt_labels), )), self.max_overlaps])

        if self.labels is not None:
            self.labels = paddle.concat([gt_labels, self.labels])


@manager.BBOX_ASSIGNERS.add_component
class HungarianAssigner3D(object):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None):
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost
        self.iou_cost = iou_cost
        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.shape[0], bbox_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = paddle.full((num_bboxes, ), -1, dtype=paddle.int64)
        assigned_labels = paddle.full((num_bboxes, ), -1, dtype=paddle.int64)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost

        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)

        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])

        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = paddle.to_tensor(matched_row_inds)
        matched_col_inds = paddle.to_tensor(matched_col_inds)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds = paddle.scatter(assigned_gt_inds, matched_row_inds,
                                          matched_col_inds + 1)
        assigned_labels = paddle.scatter(
            assigned_labels, matched_row_inds,
            paddle.gather(gt_labels, matched_col_inds, axis=0))
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
