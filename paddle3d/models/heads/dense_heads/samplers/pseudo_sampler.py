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

# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import paddle


class SamplingResult(object):
    """Bbox sampling result.
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = paddle.gather(bboxes, pos_inds)
        self.neg_bboxes = paddle.gather(bboxes, neg_inds)
        self.pos_is_gt = paddle.gather(gt_flags, pos_inds)
        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = paddle.gather(assign_result.gt_inds,
                                                  pos_inds) - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = paddle.empty_like(gt_bboxes).reshape([-1, 4])
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.reshape([-1, 4])

            self.pos_gt_bboxes = paddle.gather(
                gt_bboxes, self.pos_assigned_gt_inds.astype('int64'), axis=0)

        if assign_result.labels is not None:
            self.pos_gt_labels = paddle.gather(assign_result.labels, pos_inds)
        else:
            self.pos_gt_labels = None


class PseudoSampler(object):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (paddle.Tensor): Bounding boxes
            gt_bboxes (paddle.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = paddle.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = paddle.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = paddle.zeros([bboxes.shape[0]], dtype='int32')
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
