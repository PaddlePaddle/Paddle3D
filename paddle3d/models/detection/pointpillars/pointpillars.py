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

import numbers
import os
from collections.abc import Mapping, Sequence
from typing import List

import numpy as np
import paddle
import paddle.nn as nn
from paddle.static import InputSpec

from paddle3d.apis import manager
from paddle3d.models.base import BaseLidarModel
from paddle3d.models.detection.pointpillars.anchors_generator import \
    AnchorGenerator
from paddle3d.sample import Sample
from paddle3d.utils import checkpoint
from paddle3d.utils.logger import logger

__all__ = ["PointPillars"]


@manager.MODELS.add_component
class PointPillars(BaseLidarModel):
    def __init__(self,
                 voxelizer,
                 pillar_encoder,
                 middle_encoder,
                 backbone,
                 neck,
                 head,
                 loss,
                 anchor_configs,
                 anchor_area_threshold=1,
                 pretrained=None,
                 box_with_velocity: bool = False):
        super().__init__(
            with_voxelizer=False,
            box_with_velocity=box_with_velocity,
            max_num_points_in_voxel=pillar_encoder.max_num_points_in_voxel,
            in_channels=pillar_encoder.in_channels)

        self.voxelizer = voxelizer
        self.pillar_encoder = pillar_encoder
        self.middle_encoder = middle_encoder
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

        self.anchor_generator = AnchorGenerator(
            output_stride_factor=self.backbone.downsample_strides[0] //
            self.neck.upsample_strides[0],
            point_cloud_range=self.voxelizer.point_cloud_range,
            voxel_size=self.voxelizer.voxel_size,
            anchor_configs=anchor_configs,
            anchor_area_threshold=anchor_area_threshold)

        self.pretrained = pretrained
        self.init_weight()

    def train_forward(self, samples):
        voxels = samples["voxels"]
        coordinates = samples["coords"]
        num_points_per_voxel = samples["num_points_per_voxel"]

        # yapf: disable
        batch_size = len(samples["data"])
        pillar_features = self.pillar_encoder(
            voxels, num_points_per_voxel, coordinates)
        spatial_features = self.middle_encoder(
            pillar_features, coordinates, batch_size)
        # yapf: enable

        final_features = self.backbone(spatial_features)
        fused_final_features = self.neck(final_features)
        preds = self.head(fused_final_features)

        box_preds = preds["box_preds"]
        cls_preds = preds["cls_preds"]
        if self.head.use_direction_classifier:
            dir_preds = preds["dir_preds"]
            loss_dict = self.loss(box_preds, cls_preds, samples["reg_targets"],
                                  samples["labels"], dir_preds,
                                  self.anchor_generator.anchors)
        else:
            loss_dict = self.loss(box_preds, cls_preds, samples["reg_targets"],
                                  samples["labels"])

        return loss_dict

    def test_forward(self, samples):
        voxels = samples["voxels"]
        coordinates = samples["coords"]
        num_points_per_voxel = samples["num_points_per_voxel"]

        # yapf: disable
        batch_size = len(samples["data"])
        pillar_features = self.pillar_encoder(
            voxels, num_points_per_voxel, coordinates)
        spatial_features = self.middle_encoder(
            pillar_features, coordinates, batch_size)
        # yapf: enable

        final_features = self.backbone(spatial_features)
        fused_final_features = self.neck(final_features)
        preds = self.head(fused_final_features)

        anchors_mask = []
        for i in range(batch_size):
            batch_mask = coordinates[:, 0] == i
            this_coords = coordinates[batch_mask][:, 1:]
            anchors_mask.append(self.anchor_generator(this_coords))
        return self.head.post_process(samples, preds,
                                      self.anchor_generator.anchors,
                                      anchors_mask, batch_size)

    def export_forward(self, samples):
        voxels = samples["voxels"]
        coordinates = samples["coords"]
        num_points_per_voxel = samples["num_points_per_voxel"]

        # yapf: disable
        coordinates = paddle.concat([
            paddle.zeros([coordinates.shape[0], 1], dtype=coordinates.dtype),
            coordinates
        ], axis=-1)
        batch_size = None
        pillar_features = self.pillar_encoder(
            voxels, num_points_per_voxel, coordinates)
        spatial_features = self.middle_encoder(
            pillar_features, coordinates, batch_size)
        # yapf: enable

        final_features = self.backbone(spatial_features)
        fused_final_features = self.neck(final_features)
        preds = self.head(fused_final_features)

        anchors_mask = self.anchor_generator(coordinates[:, 1:])
        return self.head.post_process(samples, preds,
                                      self.anchor_generator.anchors,
                                      anchors_mask, batch_size)

    def init_weight(self):
        if self.pretrained is not None:
            checkpoint.load_pretrained_model(self, self.pretrained)

    def collate_fn(self, batch: List):
        sample = batch[0]
        if isinstance(sample, np.ndarray):
            batch = np.stack(batch, axis=0)
            return batch
        elif isinstance(sample, paddle.Tensor):
            return paddle.stack(batch, axis=0)
        elif isinstance(sample, numbers.Number):
            batch = np.array(batch)
            return batch
        elif isinstance(sample, (str, bytes)):
            return batch
        elif isinstance(sample, (Sample, Mapping)):
            var_len_fields = {"data", "calibs"}
            concat_fields = {"voxels", "num_points_per_voxel"}
            collated_batch = {}
            for key, value in sample.items():
                if value is None:
                    continue
                if key == "coords":
                    collated_batch[key] = np.concatenate([
                        np.pad(
                            d[key], ((0, 0), (1, 0)),
                            mode="constant",
                            constant_values=i) for i, d in enumerate(batch)
                    ])
                elif key in concat_fields:
                    collated_batch[key] = np.concatenate(
                        [d[key] for d in batch], axis=0)
                elif key not in var_len_fields or isinstance(
                        value, (Sample, Mapping)):
                    collated_batch[key] = self.collate_fn(
                        [d[key] for d in batch])
                else:
                    collated_batch[key] = [d[key] for d in batch]
            return collated_batch
        elif isinstance(sample, Sequence):
            sample_fields_num = len(sample)
            if not all(
                    len(sample) == sample_fields_num for sample in iter(batch)):
                raise RuntimeError(
                    "fileds number not same among samples in a batch")
            return [self.collate_fn(fields) for fields in zip(*batch)]

        raise TypeError(
            "batch data can only contains: tensor, numpy.ndarray, "
            "dict, list, number, paddle3d.Sample, but got {}".format(
                type(sample)))
