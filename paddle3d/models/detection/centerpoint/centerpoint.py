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

import collections
import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec

from paddle3d.apis import manager
from paddle3d.geometries import BBoxes3D
from paddle3d.models.base import BaseLidarModel
from paddle3d.sample import Sample, SampleMeta
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import logger


class DictObject(Dict):
    def __init__(self, config: Dict):
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, DictObject(value))
            else:
                setattr(self, key, value)


@manager.MODELS.add_component
class CenterPoint(BaseLidarModel):
    def __init__(self,
                 voxelizer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck,
                 bbox_head,
                 test_cfg=None,
                 pretrained=None,
                 box_with_velocity: bool = False):
        super().__init__(
            with_voxelizer=True, box_with_velocity=box_with_velocity)
        self.voxelizer = voxelizer
        self.voxel_encoder = voxel_encoder
        self.middle_encoder = middle_encoder
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head
        self.test_cfg = DictObject(test_cfg)
        self.sync_bn = True
        if pretrained is not None:
            load_pretrained_model(self, self.pretrained)

    def deploy_preprocess(self, points):
        def true_fn(points):
            points = points[:, 0:5]
            return points

        def false_fn(points):
            points = points.reshape([1, -1, 4])
            points = F.pad(
                points, [0, 1], value=0, mode='constant', data_format="NCL")
            points = points.reshape([-1, 5])
            return points

        points = paddle.static.nn.cond(
            points.shape[-1] >=
            5, lambda: true_fn(points), lambda: false_fn(points))
        return points[:, 0:self.voxel_encoder.in_channels]

    def voxelize(self, points):
        voxels, coordinates, num_points_in_voxel = self.voxelizer(points)
        return voxels, coordinates, num_points_in_voxel

    def extract_feat(self, data):
        if self.voxelizer.__class__.__name__ == 'HardVoxelizer':
            voxels, coordinates, num_points_in_voxel = self.voxelizer(
                data['points'])
            data["features"] = voxels
            data["num_points_in_voxel"] = num_points_in_voxel
            data["coors"] = coordinates
            input_features = self.voxel_encoder(
                data["features"], data["num_points_in_voxel"], data["coors"])
        elif self.voxelizer.__class__.__name__ == 'DynamicVoxelizer':
            voxels, coors = self.voxelizer(data['points'])
            input_features, feature_coors = self.voxel_encoder(voxels, coors)
            data["coors"] = feature_coors
        x = self.middle_encoder(input_features, data["coors"],
                                data["batch_size"])
        x = self.backbone(x)
        x = self.neck(x)
        return x

    def train_forward(self, samples):
        batch_size = len(samples["data"])
        points = samples["data"]

        data = dict(points=points, batch_size=batch_size)
        x = self.extract_feat(data)
        preds, x = self.bbox_head(x)

        return self.bbox_head.loss(samples, preds, self.test_cfg)

    def test_forward(self, samples):
        batch_size = len(samples["data"])
        points = samples["data"]

        data = dict(points=points, batch_size=batch_size)
        x = self.extract_feat(data)
        preds, x = self.bbox_head(x)

        preds = self.bbox_head.predict_by_custom_op(samples, preds,
                                                    self.test_cfg)
        preds = self._parse_results_to_sample(preds, samples)
        return {'preds': preds}

    def export_forward(self, samples):
        batch_size = 1
        points = samples["data"]
        points = self.deploy_preprocess(points)

        data = dict(points=points, batch_size=batch_size)
        x = self.extract_feat(data)
        preds, x = self.bbox_head(x)

        return self.bbox_head.predict_by_custom_op(samples, preds,
                                                   self.test_cfg)

    def _parse_results_to_sample(self, results: dict, sample: dict):
        num_samples = len(results)
        new_results = []
        for i in range(num_samples):
            data = Sample(sample["path"][i], sample["modality"][i])
            bboxes_3d = results[i]["box3d_lidar"].numpy()
            labels = results[i]["label_preds"].numpy()
            confidences = results[i]["scores"].numpy()
            data.bboxes_3d = BBoxes3D(bboxes_3d[:, [0, 1, 2, 3, 4, 5, -1]])
            data.bboxes_3d.coordmode = 'Lidar'
            data.bboxes_3d.origin = [0.5, 0.5, 0.5]
            data.bboxes_3d.rot_axis = 2
            if bboxes_3d.shape[-1] == 9:
                data.bboxes_3d.velocities = bboxes_3d[:, 6:8]
            data.labels = labels
            data.confidences = confidences
            data.meta = SampleMeta(id=results[i]["meta"])
            if "calibs" in sample:
                calib = [calibs.numpy()[i] for calibs in sample["calibs"]]
                data.calibs = calib
            new_results.append(data)
        return new_results

    def collate_fn(self, batch: List):
        """
        """
        sample_merged = collections.defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                sample_merged[k].append(v)
        batch_size = len(sample_merged['meta'])
        ret = {}
        for key, elems in sample_merged.items():
            if key in ["voxels", "num_points_per_voxel"]:
                ret[key] = np.concatenate(elems, axis=0)
            elif key in ["meta"]:
                ret[key] = [elem.id for elem in elems]
            elif key in ["path", "modality"]:
                ret[key] = elems
            elif key == "data":
                ret[key] = [elem for elem in elems]
            elif key == "coords":
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = np.pad(
                        coor, ((0, 0), (1, 0)),
                        mode="constant",
                        constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in [
                    "heat_map", "target_bbox", "center_idx", "target_mask",
                    "target_label", "calibs"
            ]:
                ret[key] = collections.defaultdict(list)
                res = []
                for elem in elems:
                    for idx, ele in enumerate(elem):
                        ret[key][str(idx)].append(ele)
                for kk, vv in ret[key].items():
                    res.append(np.stack(vv))
                ret[key] = res
        return ret
