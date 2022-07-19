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

import os

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.models import layers
from paddle3d.ops import iou3d_nms_cuda


def sigmoid_hm(hm_features):
    """sigmoid to headmap

    Args:
        hm_features (paddle.Tensor): heatmap

    Returns:
        paddle.Tensor: sigmoid heatmap
    """
    x = F.sigmoid(hm_features)
    x = x.clip(min=1e-4, max=1 - 1e-4)

    return x


def nms_hm(heat_map, kernel=3):
    """Do max_pooling for nms

    Args:
        heat_map (paddle.Tensor): pred cls heatmap
        kernel (int, optional): max_pool kernel size. Defaults to 3.

    Returns:
        heatmap after nms
    """
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        heat_map, kernel_size=(kernel, kernel), stride=1, padding=pad)
    eq_index = (hmax == heat_map).astype("float32")

    return heat_map * eq_index


def select_topk(heat_map, K=100):
    """
    Args:
        heat_map: heat_map in [N, C, H, W]
        K: top k samples to be selected
        score: detection threshold

    Returns:

    """

    #batch, c, height, width = paddle.shape(heat_map)

    batch, c = heat_map.shape[:2]
    height = paddle.shape(heat_map)[2]
    width = paddle.shape(heat_map)[3]

    # First select topk scores in all classes and batchs
    # [N, C, H, W] -----> [N, C, H*W]
    heat_map = paddle.reshape(heat_map, (batch, c, -1))
    # Both in [N, C, K]
    topk_scores_all, topk_inds_all = paddle.topk(heat_map, K)

    # topk_inds_all = topk_inds_all % (height * width) # todo: this seems redudant
    topk_ys = (topk_inds_all // width).astype("float32")
    topk_xs = (topk_inds_all % width).astype("float32")

    # Select topK examples across channel
    # [N, C, K] -----> [N, C*K]
    topk_scores_all = paddle.reshape(topk_scores_all, (batch, -1))
    # Both in [N, K]
    topk_scores, topk_inds = paddle.topk(topk_scores_all, K)
    topk_clses = (topk_inds // K).astype("float32")

    # First expand it as 3 dimension
    topk_inds_all = paddle.reshape(
        _gather_feat(paddle.reshape(topk_inds_all, (batch, -1, 1)), topk_inds),
        (batch, K))
    topk_ys = paddle.reshape(
        _gather_feat(paddle.reshape(topk_ys, (batch, -1, 1)), topk_inds),
        (batch, K))
    topk_xs = paddle.reshape(
        _gather_feat(paddle.reshape(topk_xs, (batch, -1, 1)), topk_inds),
        (batch, K))

    return dict({
        "topk_score": topk_scores,
        "topk_inds_all": topk_inds_all,
        "topk_clses": topk_clses,
        "topk_ys": topk_ys,
        "topk_xs": topk_xs
    })


def _gather_feat(feat, ind, mask=None):
    """Select specific indexs on featuremap

    Args:
        feat: all results in 3 dimensions
        ind: positive index

    Returns:

    """
    channel = feat.shape[-1]
    ind = ind.unsqueeze(-1).expand((ind.shape[0], ind.shape[1], channel))

    feat = gather(feat, ind)

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, channel)
    return feat


def _transpose_and_gather_feat(feat, ind):
    def _gather_feat(feat, ind, mask=None):
        dim = feat.shape[2]
        ind = ind.unsqueeze(2)
        bs_ind = paddle.arange(ind.shape[0], dtype=ind.dtype)
        bs_ind = paddle.tile(bs_ind, repeat_times=[1, ind.shape[1], 1])
        bs_ind = bs_ind.transpose([2, 1, 0])
        ind = paddle.concat([bs_ind, ind], axis=-1)
        feat = feat.gather_nd(ind)
        feat = feat.reshape(feat.shape[0:2] + [dim])
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.reshape([-1, dim])
        return feat

    feat = feat.transpose([0, 2, 3, 1])
    feat = feat.reshape([feat.shape[0], -1, feat.shape[3]])
    feat = _gather_feat(feat, ind)
    return feat


def gather(feature: paddle.Tensor, ind: paddle.Tensor):
    """Simplified version of torch.gather. Always gather based on axis 1.

    Args:
        feature: all results in 3 dimensions, such as [n, h * w, c]
        ind: positive index in 3 dimensions, such as [n, k, 1]

    Returns:
        gather feature
    """
    bs_ind = paddle.arange(ind.shape[0], dtype=ind.dtype)
    bs_ind = bs_ind.unsqueeze(1).unsqueeze(2)
    bs_ind = bs_ind.expand([ind.shape[0], ind.shape[1], 1])
    ind = paddle.concat([bs_ind, ind], axis=-1)

    return feature.gather_nd(ind)


def select_point_of_interest(batch, index, feature_maps):
    """
    Select POI(point of interest) on feature map
    Args:
        batch: batch size
        index: in point format or index format
        feature_maps: regression feature map in [N, C, H, W]

    Returns:

    """
    w = feature_maps.shape[3]
    index_length = len(index.shape)
    if index_length == 3:
        index = index[:, :, 1] * w + index[:, :, 0]
    index = paddle.reshape(index, (batch, -1))
    # [N, C, H, W] -----> [N, H, W, C]
    feature_maps = paddle.transpose(feature_maps, (0, 2, 3, 1))
    channel = feature_maps.shape[-1]
    # [N, H, W, C] -----> [N, H*W, C]
    feature_maps = paddle.reshape(feature_maps, (batch, -1, channel))
    index = index.unsqueeze(-1)
    # select specific features bases on POIs

    feature_maps = gather(feature_maps, index)

    return feature_maps


def rotate_nms_pcdet(boxes,
                     scores,
                     thresh,
                     pre_max_size=None,
                     post_max_size=None):
    """
    :param boxes: (N, 5) [x, y, z, l, w, h, theta]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # transform back to pcdet's coordinate
    index = paddle.to_tensor([0, 1, 2, 4, 3, 5, boxes.shape[-1] - 1],
                             dtype='int32')

    boxes = paddle.index_select(boxes, index=index, axis=-1)
    #boxes = boxes[:, [0, 1, 2, 4, 3, 5, -1]]

    boxes[:, -1] = -boxes[:, -1] - np.pi / 2

    order = scores.argsort(0, descending=True)
    if pre_max_size is not None:
        order = order[:pre_max_size]

    boxes = boxes[order]
    # TODO(luoqianhui): when order.shape is (1,),
    # boxes[order].shape is (7,) but supposed to be (1, 7),
    # so we add a reshape op
    boxes = boxes.reshape([-1, 7])

    keep, num_out = iou3d_nms_cuda.nms_gpu(boxes, thresh)
    selected = order[keep[:num_out]]

    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = nn.BatchNorm2D(out_channels, data_format=data_format)
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = nn.BatchNorm2D(out_channels, data_format=data_format)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class SeparableConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 pointwise_bias=None,
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self.piontwise_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            data_format=data_format,
            bias_attr=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x
