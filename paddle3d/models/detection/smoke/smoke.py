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
from typing import List, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.geometries import BBoxes2D, BBoxes3D, CoordMode
from paddle3d.models.base import BaseMonoModel
from paddle3d.models.detection.smoke.processor import PostProcessor
from paddle3d.models.detection.smoke.smoke_loss import SMOKELossComputation
from paddle3d.sample import Sample
from paddle3d.utils.logger import logger


@manager.MODELS.add_component
class SMOKE(BaseMonoModel):
    """
    """

    def __init__(self,
                 backbone,
                 head,
                 depth_ref: Tuple,
                 dim_ref: Tuple,
                 max_detection: int = 50,
                 pred_2d: bool = True,
                 box_with_velocity: bool = False):
        super().__init__(
            box_with_velocity=box_with_velocity,
            need_camera_to_image=True,
            need_lidar_to_camera=False,
            need_down_ratios=True)

        self.backbone = backbone
        self.heads = head
        self.max_detection = max_detection

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
        images = samples['images']
        features = self.backbone(images)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        predictions = self.heads(features)
        return self.post_process.export_forward(
            predictions, [samples['trans_cam_to_img'], samples['down_ratios']])

    def train_forward(self, samples):
        images = samples['data']
        features = self.backbone(images)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        predictions = self.heads(features)
        loss = self.loss_computation(predictions, samples['target'])
        return {'loss': loss}

    def test_forward(self, samples):
        images = samples['data']
        features = self.backbone(images)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        predictions = self.heads(features)
        bs = predictions[0].shape[0]
        predictions = self.post_process(predictions, samples['target'])
        res = [
            self._parse_results_to_sample(predictions, samples, i)
            for i in range(bs)
        ]
        return {'preds': res}

    def init_weight(self, bias_lr_factor=2):
        for sublayer in self.sublayers():
            if hasattr(sublayer, 'bias') and sublayer.bias is not None:
                sublayer.bias.optimize_attr['learning_rate'] = bias_lr_factor

    def _parse_results_to_sample(self, results: paddle.Tensor, sample: dict,
                                 index: int):
        ret = Sample(sample['path'][index], sample['modality'][index])
        ret.meta.update(
            {key: value[index]
             for key, value in sample['meta'].items()})

        if 'calibs' in sample:
            ret.calibs = [
                sample['calibs'][i][index]
                for i in range(len(sample['calibs']))
            ]

        if results.shape[0] != 0:
            results = results[results[:, 14] == index][:, :14]
            results = results.numpy()
            clas = results[:, 0]
            bboxes_2d = BBoxes2D(results[:, 2:6])

            # TODO: fix hard code here
            bboxes_3d = BBoxes3D(
                results[:, [9, 10, 11, 8, 6, 7, 12]],
                coordmode=CoordMode.KittiCamera,
                origin=(0.5, 1, 0.5),
                rot_axis=1)

            confidences = results[:, 13]

            ret.confidences = confidences
            ret.bboxes_2d = bboxes_2d
            ret.bboxes_3d = bboxes_3d
            ret.labels = clas

        return ret

    @property
    def inputs(self) -> List[dict]:
        images = {
            'name': 'images',
            'dtype': 'float32',
            'shape': [1, 3, self.image_height, self.image_width]
        }
        res = [images]

        intrinsics = {
            'name': 'trans_cam_to_img',
            'dtype': 'float32',
            'shape': [1, 3, 3]
        }
        res.append(intrinsics)

        down_ratios = {
            'name': 'down_ratios',
            'dtype': 'float32',
            'shape': [1, 2]
        }
        res.append(down_ratios)
        return res

    @property
    def outputs(self) -> List[dict]:
        data = {
            'name': 'smoke_output',
            'dtype': 'float32',
            'shape': [self.max_detection, 14]
        }
        return [data]
