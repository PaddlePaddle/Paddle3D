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

import math
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.models.base import add_export_args
from paddle3d.models.layers import param_init
from paddle3d.sample import Sample
from paddle3d.utils import checkpoint
from paddle3d.utils.logger import logger

__all__ = ["SqueezeSegV3"]


@manager.MODELS.add_component
class SqueezeSegV3(nn.Layer):
    """
    The SqueezeSegV3 implementation based on PaddlePaddle.

    Please refer to:
        Xu, Chenfeng, et al. “SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation.”
        CoRR, vol. abs/2004.01803, 2020, https://arxiv.org/abs/2004.01803.

    Args:
        backbone (paddle.nn.Layer): Backbone network.
        loss (paddle.nn.Layer): Loss layer.
        num_classes (int): Number of classes.
        pretrained (str): Path to pretrained model.
    """

    def __init__(self,
                 backbone: paddle.nn.Layer,
                 loss: paddle.nn.Layer,
                 num_classes: int = 20,
                 pretrained: str = None):
        super().__init__()

        self.backbone = backbone
        self.loss = loss

        self.heads = nn.LayerList([
            nn.Conv2D(256, num_classes, 1, padding=0),
            nn.Conv2D(256, num_classes, 1, padding=0),
            nn.Conv2D(128, num_classes, 1, padding=0),
            nn.Conv2D(64, num_classes, 1, padding=0),
            nn.Conv2D(32, num_classes, 3, padding=1)
        ])

        self.pretrained = pretrained
        self.init_weight()

        self.sync_bn = True

    def forward(self, samples):
        range_images = paddle.stack(samples["data"], axis=0)
        feature_list = self.backbone(range_images)

        if self.training:
            logits_list = []
            for head, feat in zip(self.heads, feature_list):
                logits = head(feat)
                logits = F.softmax(logits, axis=1)
                logits_list.append(logits)
            loss = self.loss(logits_list, paddle.stack(
                samples['labels'], axis=0))
            return {"loss": loss}
        else:
            # TODO(will-jl944): support multi-card evaluation and prediction
            logits = self.heads[-1](feature_list[-1])
            prediction = paddle.argmax(logits, axis=1)

            # de-batchify
            ret = []
            for batch_idx, pred in enumerate(prediction):
                sample = Sample(
                    path=samples["path"][batch_idx],
                    modality=samples["modality"][batch_idx])
                sample.labels = pred[samples["meta"]["proj_y"][batch_idx],
                                     samples["meta"]["proj_x"][batch_idx]]
                ret.append(sample)

            return {"preds": ret}

    def init_weight(self):
        if self.pretrained is not None:
            checkpoint.load_pretrained_model(self, self.pretrained)
        else:
            for layer in self.sublayers():
                if isinstance(layer, (nn.Conv2D, nn.Conv2DTranspose)):
                    param_init.kaiming_uniform_init(
                        layer.weight, a=math.sqrt(5))
                    if layer.bias is not None:
                        fan_in, _ = param_init._calculate_fan_in_and_fan_out(
                            layer.weight)
                        if fan_in != 0:
                            bound = 1 / math.sqrt(fan_in)
                            param_init.uniform_init(layer.bias, -bound, bound)

    def export_forward(self, samples):
        range_images = samples
        feature_list = self.backbone(range_images)
        logits = self.heads[-1](feature_list[-1])
        prediction = paddle.argmax(logits, axis=1)

        return prediction

    @add_export_args('--input_shape', nargs='+', type=int, required=True)
    def export(self, save_dir: str, input_shape: list = None, **kwargs):
        self.forward = self.export_forward
        save_path = os.path.join(save_dir, 'squeezesegv3')
        if input_shape is None:
            raise ValueError("input_shape must be provided!")
        elif len(input_shape) == 1:
            shape = [
                None, self.backbone.in_channels, input_shape[0], input_shape[0]
            ]
        elif len(input_shape) == 2:
            shape = [None, self.backbone.in_channels] + input_shape
        else:
            shape = input_shape
        input_spec = [paddle.static.InputSpec(shape=shape, dtype="float32")]

        paddle.jit.to_static(self, input_spec=input_spec)
        paddle.jit.save(self, save_path)

        logger.info("Exported model is saved in {}".format(save_dir))
