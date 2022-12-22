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

from paddle3d.ops import iou3d_nms_cuda


def class_agnostic_nms(box_scores,
                       box_preds,
                       label_preds,
                       nms_config,
                       score_thresh=None):
    def nms(box_scores, box_preds, label_preds, nms_config):
        order = box_scores.argsort(0, descending=True)
        order = order[:nms_config['nms_pre_maxsize']]
        box_preds = paddle.gather(box_preds, index=order)
        box_scores = paddle.gather(box_scores, index=order)
        label_preds = paddle.gather(label_preds, index=order)
        # When order is one-value tensor,
        # boxes[order] loses a dimension, so we add a reshape
        keep, num_out = iou3d_nms_cuda.nms_gpu(box_preds,
                                               nms_config['nms_thresh'])
        selected = keep[0:num_out]
        selected = selected[:nms_config['nms_post_maxsize']]
        selected_score = paddle.gather(box_scores, index=selected)
        selected_box = paddle.gather(box_preds, index=selected)
        selected_label = paddle.gather(label_preds, index=selected)
        return selected_score, selected_label, selected_box

    if score_thresh is not None:
        scores_mask = box_scores >= score_thresh

        def box_empty(box_scores, box_preds, label_preds):
            fake_score = paddle.to_tensor([-1.0], dtype=box_scores.dtype)
            fake_label = paddle.to_tensor([-1.0], dtype=label_preds.dtype)
            fake_box = paddle.to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                                        dtype=box_preds.dtype)

            return fake_score, fake_label, fake_box

        def box_not_empty(scores_mask, box_scores, box_preds, label_preds,
                          nms_config):
            nonzero_index = paddle.nonzero(scores_mask)
            box_scores = paddle.gather(box_scores, index=nonzero_index)
            box_preds = paddle.gather(box_preds, index=nonzero_index)
            label_preds = paddle.gather(label_preds, index=nonzero_index)
            return nms(box_scores, box_preds, label_preds, nms_config)

        return paddle.static.nn.cond(
            paddle.logical_not(scores_mask.any()), lambda: box_empty(
                box_scores, box_preds, label_preds), lambda: box_not_empty(
                    scores_mask, box_scores, box_preds, label_preds, nms_config)
        )
    else:
        return nms(box_scores, box_preds, label_preds, nms_config)
