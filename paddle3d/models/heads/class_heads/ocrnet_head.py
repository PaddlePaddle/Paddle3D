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
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models import layers
from paddle3d.models.layers import param_init, reset_parameters


@manager.HEADS.add_component
class OCRHead(nn.Layer):
    """
    The Object contextual representation head.
    Args:
        num_classes(int): The unique number of target classes.
        in_channels(tuple): The number of input channels.
        ocr_mid_channels(int, optional): The number of middle channels in OCRHead. Default: 512.
        ocr_key_channels(int, optional): The number of key channels in ObjectAttentionBlock. Default: 256.
    """

    def __init__(self,
                 num_classes,
                 backbone_indices,
                 in_channels,
                 ocr_mid_channels=512,
                 ocr_key_channels=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        self.in_channels = in_channels
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.spatial_gather = SpatialGatherBlock(ocr_mid_channels, num_classes)
        self.spatial_ocr = SpatialOCRModule(ocr_mid_channels, ocr_key_channels,
                                            ocr_mid_channels)

        self.indices = [-2, -1] if len(in_channels) > 1 else [-1, -1]

        self.conv3x3_ocr = layers.ConvBNReLU(
            in_channels[self.indices[1]], ocr_mid_channels, 3, padding=1)
        self.cls_head = nn.Conv2D(ocr_mid_channels, self.num_classes, 1)
        self.aux_head = nn.Sequential(
            layers.ConvBNReLU(in_channels[self.indices[0]],
                              in_channels[self.indices[0]], 1),
            nn.Conv2D(in_channels[self.indices[0]], self.num_classes, 1))

        self.init_weight()

    def forward(self, feat_list, img_shape):
        feat_list = [feat_list[i] for i in self.backbone_indices]
        feat_shallow, feat_deep = feat_list[self.indices[0]], feat_list[
            self.indices[1]]

        soft_regions = self.aux_head(feat_shallow)
        pixels = self.conv3x3_ocr(feat_deep)

        object_regions = self.spatial_gather(pixels, soft_regions)
        ocr = self.spatial_ocr(pixels, object_regions)

        logit = self.cls_head(ocr)
        logit_list = [logit, soft_regions]

        logit_list = logit_list[0]

        return logit_list

    def init_weight(self):
        """Initialize the parameters of model parts."""
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                reset_parameters(sublayer)


class SpatialGatherBlock(nn.Layer):
    """Aggregation layer to compute the pixel-region representation."""

    def __init__(self, pixels_channels, regions_channels):
        super().__init__()
        self.pixels_channels = pixels_channels
        self.regions_channels = regions_channels

    def forward(self, pixels, regions):
        # pixels: from (n, c, h, w) to (n, h*w, c)
        pixels = paddle.reshape(pixels, (0, self.pixels_channels, -1))
        pixels = paddle.transpose(pixels, (0, 2, 1))

        # regions: from (n, k, h, w) to (n, k, h*w)
        regions = paddle.reshape(regions, (0, self.regions_channels, -1))
        regions = F.softmax(regions, axis=2)

        # feats: from (n, k, c) to (n, c, k, 1)
        feats = paddle.bmm(regions, pixels)
        feats = paddle.transpose(feats, (0, 2, 1))
        feats = paddle.unsqueeze(feats, axis=-1)

        return feats


class SpatialOCRModule(nn.Layer):
    """Aggregate the global object representation to update the representation for each pixel."""

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 dropout_rate=0.1):
        super().__init__()

        self.attention_block = ObjectAttentionBlock(in_channels, key_channels)
        self.conv1x1 = nn.Sequential(
            layers.ConvBNReLU(2 * in_channels, out_channels, 1),
            nn.Dropout2D(dropout_rate))

    def forward(self, pixels, regions):
        context = self.attention_block(pixels, regions)
        feats = paddle.concat([context, pixels], axis=1)
        feats = self.conv1x1(feats)

        return feats


class ObjectAttentionBlock(nn.Layer):
    """A self-attention module."""

    def __init__(self, in_channels, key_channels):
        super().__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels

        self.f_pixel = nn.Sequential(
            layers.ConvBNReLU(in_channels, key_channels, 1),
            layers.ConvBNReLU(key_channels, key_channels, 1))

        self.f_object = nn.Sequential(
            layers.ConvBNReLU(in_channels, key_channels, 1),
            layers.ConvBNReLU(key_channels, key_channels, 1))

        self.f_down = layers.ConvBNReLU(in_channels, key_channels, 1)

        self.f_up = layers.ConvBNReLU(key_channels, in_channels, 1)

    def forward(self, x, proxy):
        x_shape = paddle.shape(x)
        # query : from (n, c1, h1, w1) to (n, h1*w1, key_channels)
        query = self.f_pixel(x)
        query = paddle.reshape(query, (0, self.key_channels, -1))
        query = paddle.transpose(query, (0, 2, 1))

        # key : from (n, c2, h2, w2) to (n, key_channels, h2*w2)
        key = self.f_object(proxy)
        key = paddle.reshape(key, (0, self.key_channels, -1))

        # value : from (n, c2, h2, w2) to (n, h2*w2, key_channels)
        value = self.f_down(proxy)
        value = paddle.reshape(value, (0, self.key_channels, -1))
        value = paddle.transpose(value, (0, 2, 1))

        # sim_map (n, h1*w1, h2*w2)
        sim_map = paddle.bmm(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)

        # context from (n, h1*w1, key_channels) to (n , out_channels, h1, w1)
        context = paddle.bmm(sim_map, value)
        context = paddle.transpose(context, (0, 2, 1))
        context = paddle.reshape(context,
                                 (0, self.key_channels, x_shape[2], x_shape[3]))
        context = self.f_up(context)

        return context
