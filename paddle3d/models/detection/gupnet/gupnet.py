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

import paddle
import paddle.nn as nn
import numpy as np
from typing import List
from paddle3d.apis import manager
from paddle3d.models.detection.gupnet.gupnet_dla import GUP_DLA34
from paddle3d.models.detection.gupnet.gupnet_processor import GUPNETPostProcessor
from paddle3d.models.detection.gupnet.gupnet_predictor import GUPNETPredictor
from paddle3d.models.detection.gupnet.gupnet_loss import GUPNETLoss, Hierarchical_Task_Learning
from paddle3d.models.base import BaseMonoModel
from paddle3d.geometries import BBoxes2D, BBoxes3D, CoordMode
from paddle3d.sample import Sample


@manager.MODELS.add_component
class GUPNET(BaseMonoModel):
    """
    """

    def __init__(self,
                 backbone,
                 head,
                 max_detection: int = 50,
                 threshold=0.2,
                 stat_epoch_nums=5,
                 max_epoch=140,
                 train_datasets_length=3712):
        super(GUPNET, self).__init__()

        self.max_detection = max_detection
        self.train_datasets_length = train_datasets_length
        mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                              [1.52563191462, 1.62856739989, 3.88311640418],
                              [1.73698127, 0.59706367, 1.76282397]])
        self.mean_size = paddle.to_tensor(mean_size, dtype=paddle.float32)
        self.cls_num = self.mean_size.shape[0]
        self.backbone = backbone
        self.head = head
        self.loss = GUPNETLoss()
        self.ei_loss = {
            'seg_loss': paddle.to_tensor(110.),
            'offset2d_loss': paddle.to_tensor(1.6),
            'size2d_loss': paddle.to_tensor(30.),
            'depth_loss': paddle.to_tensor(8.5),
            'offset3d_loss': paddle.to_tensor(0.6),
            'size3d_loss': paddle.to_tensor(0.7),
            'heading_loss': paddle.to_tensor(3.6)
        }
        self.cur_loss = paddle.zeros(paddle.to_tensor(1))
        self.cur_loss_weightor = Hierarchical_Task_Learning(
            self.ei_loss, stat_epoch_nums=stat_epoch_nums, max_epoch=max_epoch)
        self.post_processor = GUPNETPostProcessor(mean_size, threshold)

    # TODO: fix export function
    def export_forward(self, samples):
        images = samples['images']
        features = self.backbone(images)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        predictions = self.heads(features)
        return self.post_process.export_forward(
            predictions, [samples['trans_cam_to_img'], samples['down_ratios']])

    def train_forward(self, samples):
        # encode epoch
        if not hasattr(self, 'cur_epoch'):
            self.cur_epoch = 1
            self.pre_epoches = 1
            self.loss_weights = {}  # 初始化loss权重

        input, calibs_p2, coord_ranges, targets, info, sample = samples

        if not hasattr(self, 'have_load_img_ids'):
            self.have_load_img_ids = info['img_id']
            self.trained_batch = 1
            self.stat_dict = {}

        elif info['img_id'][0] not in self.have_load_img_ids:
            self.have_load_img_ids = paddle.concat((self.have_load_img_ids,
                                                    info['img_id']))
            self.trained_batch += 1
        else:
            self.cur_epoch += 1
            del self.have_load_img_ids
            del self.trained_batch
            del self.stat_dict
            self.have_load_img_ids = info['img_id']
            self.trained_batch = 1
            self.stat_dict = {}

        feat = self.backbone(input)
        ret = self.head(feat, targets, calibs_p2, coord_ranges, is_train=True)

        loss_terms = self.loss(ret, targets)

        if not self.loss_weights:
            self.loss_weights = self.cur_loss_weightor.compute_weight(
                self.ei_loss, self.cur_epoch)
        elif self.cur_epoch != self.pre_epoches:
            self.loss_weights = self.cur_loss_weightor.compute_weight(
                self.ei_loss, self.cur_epoch)
            self.pre_epoches += 1

        # update loss with loss_weights
        loss = paddle.zeros(paddle.to_tensor(1))
        for key in self.loss_weights.keys():
            loss += self.loss_weights[key].detach() * loss_terms[key]

        # accumulate statistics
        for key in loss_terms.keys():
            if key not in self.stat_dict.keys():
                self.stat_dict[key] = 0
            self.stat_dict[key] += loss_terms[key]

        if len(self.have_load_img_ids) == self.train_datasets_length:
            for key in self.stat_dict.keys():
                self.stat_dict[key] /= self.trained_batch
            self.ei_loss = self.stat_dict

        return {'loss': loss}

    def test_forward(self, samples):
        input, calibs_p2, coord_ranges, targets, info, sample = samples
        feat = self.backbone(input)
        ret = self.head(feat, targets, calibs_p2, coord_ranges, is_train=False)
        predictions = self.post_processor(ret, info, calibs_p2)

        res = []
        for id, img_id in enumerate(predictions.keys()):
            res.append(
                self._parse_results_to_sample(predictions[img_id], sample, id))

        return {'preds': res}

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

        if len(results):
            results = paddle.to_tensor(results)

            results = results.numpy()
            clas = results[:, 0]
            bboxes_2d = BBoxes2D(results[:, 2:6])

            bboxes_3d = BBoxes3D(
                results[:, [9, 10, 11, 8, 6, 7, 12]],
                coordmode=CoordMode.KittiCamera,
                origin=(0.5, 1, 0.5),
                rot_axis=1)

            confidences = results[:, 13]

            ret.confidences = confidences
            ret.bboxes_2d = bboxes_2d
            ret.bboxes_3d = bboxes_3d

            for i in range(len(clas)):
                clas[i] = clas[i] - 1 if clas[i] > 0 else 2
            ret.labels = clas

        return ret

    def init_weight(self, layers):
        for m in layers.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
            'name': 'gupnet_output',
            'dtype': 'float32',
            'shape': [self.max_detection, 14]
        }
        return [data]
