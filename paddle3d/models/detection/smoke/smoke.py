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
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.geometries import BBoxes2D, BBoxes3D, CoordMode
from paddle3d.models.detection.smoke.processor import PostProcessor
from paddle3d.models.detection.smoke.smoke_loss import SMOKELossComputation
from paddle3d.sample import Sample
from paddle3d.utils.logger import logger


@manager.MODELS.add_component
class SMOKE(nn.Layer):
    """
    """

    def __init__(self,
                 backbone,
                 head,
                 depth_ref: Tuple,
                 dim_ref: Tuple,
                 max_detection: int = 50,
                 pred_2d: bool = True):
        super().__init__()
        self.backbone = backbone
        self.heads = head
        self.init_weight()
        self.loss_computation = SMOKELossComputation(
            depth_ref=depth_ref,
            dim_ref=dim_ref,
            reg_loss="DisL1",
            loss_weight=[1., 10.],
            max_objs=max_detection)

        self.post_process = PostProcessor(
            depth_ref=depth_ref,
            dim_ref=dim_ref,
            reg_head=self.heads.reg_heads,
            max_detection=max_detection,
            pred_2d=pred_2d)

    def export_forward(self, samples):
        images = samples[0]
        features = self.backbone(images)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        predictions = self.heads(features)
        return self.post_process.export_forward(predictions, samples[1:])

    def forward(self, samples):
        images = samples['data']
        features = self.backbone(images)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        predictions = self.heads(features)
        if not self.training:
            # TODO: Inefficient temporary solution, fix this by perform batched post-processing
            res = []
            bs = predictions[0].shape[0]
            for i in range(bs):
                inputs = [
                    predictions[0][i].unsqueeze(0),
                    predictions[1][i].unsqueeze(0)
                ]
                prediction = self.post_process(inputs, samples['target'])
                res.append(
                    self._parse_results_to_sample(prediction, samples, i))

            return {'preds': res}

        loss = self.loss_computation(predictions, samples['target'])
        return {'loss': loss}

    def init_weight(self, bias_lr_factor=2):
        for sublayer in self.sublayers():
            if hasattr(sublayer, 'bias') and sublayer.bias is not None:
                sublayer.bias.optimize_attr['learning_rate'] = bias_lr_factor

    def _parse_results_to_sample(self, results: paddle.Tensor, sample: dict,
                                 index: int):
        ret = sample.copy()
        results = results.numpy()

        if results.shape[0] != 0:
            clas = results[:, 0]
            bboxes_2d = BBoxes2D(results[:, 2:6])

            # TODO: fix hard code here
            bboxes_3d = BBoxes3D(
                results[:, 6:13],
                coordmode=CoordMode.KittiCamera,
                origin=(0.5, 1, 0.5),
                rot_axis=1)

            confidences = results[:, 13]

            ret.confidences = confidences
            ret.bboxes_2d = bboxes_2d
            ret.bboxes_3d = bboxes_3d
            ret.labels = clas

        return ret

    def export(self, save_dir: str, **kwargs):
        self.forward = self.export_forward
        image_spec = paddle.static.InputSpec(
            shape=[1, 3, None, None], dtype="float32")
        K_spec = paddle.static.InputSpec(shape=[1, 3, 3], dtype="float32")
        down_ratio_spec = paddle.static.InputSpec(shape=[1, 2], dtype="float32")

        input_spec = [[image_spec, K_spec, down_ratio_spec]]

        paddle.jit.to_static(self, input_spec=input_spec)
        paddle.jit.save(self, os.path.join(save_dir, "inference"))
