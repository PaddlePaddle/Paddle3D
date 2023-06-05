# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn.functional as F
import paddle


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = F.max_pool2d(
        heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).astype('float32')
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.shape

    # batch * cls_ids * 50
    topk_scores, topk_inds = paddle.topk(heatmap.reshape((batch, cat, -1)), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).astype('int32').astype('float32')
    topk_xs = (topk_inds % width).astype('int32').astype('float32')

    # batch * cls_ids * 50
    topk_score, topk_ind = paddle.topk(topk_scores.reshape((batch, -1)), K)
    topk_cls_ids = (topk_ind / K).astype('int32')
    topk_inds = _gather_feat(topk_inds.reshape((batch, -1, 1)),
                             topk_ind).reshape((batch, K))
    topk_ys = _gather_feat(topk_ys.reshape((batch, -1, 1)), topk_ind).reshape(
        (batch, K))
    topk_xs = _gather_feat(topk_xs.reshape((batch, -1, 1)), topk_ind).reshape(
        (batch, K))

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim = feat.shape[2]  # get channel dim
    # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    ind = ind.unsqueeze(2).expand(shape=[ind.shape[0], ind.shape[1], dim])
    feat = paddle.take_along_axis(feat, indices=ind, axis=1)

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.transpose(perm=(0, 2, 3, 1))  # B * C * H * W ---> B * H * W * C
    feat = feat.reshape((feat.shape[0], -1,
                         feat.shape[3]))  # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)  # B * len(ind) * C
    return feat


def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]
