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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import List

import numpy as np
import paddle

from paddle3d.apis import manager
from paddle3d.datasets.semantic_kitti.semantic_kitti import \
    SemanticKITTIDataset
from paddle3d.sample import Sample
from paddle3d.utils.logger import logger

from .semantic_kitti_metric import SemanticKITTIMetric

__all__ = ["SemanticKITTISegDataset"]


@manager.DATASETS.add_component
class SemanticKITTISegDataset(SemanticKITTIDataset):
    """
    SemanticKITTI dataset for semantic segmentation task.
    """

    def __getitem__(self, index: int) -> Sample:
        sample = Sample(path=self.data[index], modality="lidar")

        if not self.is_test_mode:
            scan_path = Path(self.data[index])
            label_path = (scan_path.parents[1] / "labels" /
                          scan_path.name).with_suffix(".label")
            sample.labels = label_path

        if self.transforms:
            sample = self.transforms(sample)

        if "proj_mask" in sample.meta:
            sample.data *= sample.meta.pop("proj_mask")

        return sample

    def collate_fn(self, batch: List):
        """
        """
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
        elif isinstance(sample, Sample):
            var_len_fields = {"data", "labels", "proj_x", "proj_y"}
            collated_batch = {}
            for key, value in sample.items():
                if value is None:
                    continue
                if key not in var_len_fields or isinstance(
                        value, (Sample, Mapping)):
                    collated_batch[key] = self.collate_fn(
                        [d[key] for d in batch])
                else:
                    collated_batch[key] = [d[key] for d in batch]
            return collated_batch
        elif isinstance(sample, Mapping):
            var_len_fields = {"data", "labels", "proj_x", "proj_y"}
            collated_batch = {}
            for key, value in sample.items():
                if key not in var_len_fields or isinstance(
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

    @property
    def metric(self):
        ignore = []
        for cl, ign in self.LEARNING_IGNORE.items():
            if ign:
                x_cl = int(cl)
                ignore.append(x_cl)
                logger.info(
                    "Cross-entropy class {} ignored in IoU evaluation".format(
                        x_cl))
        return SemanticKITTIMetric(len(self.LEARNING_MAP_INV), ignore=ignore)
