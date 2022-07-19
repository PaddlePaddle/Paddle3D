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
from typing import List

import numpy as np
import paddle
import paddle.fluid.layers as layers

from paddle3d.geometries import BBoxes2D, BBoxes3D
from paddle3d.sample import Sample


class BaseDataset(paddle.io.Dataset):
    """
    """

    @property
    def is_train_mode(self) -> bool:
        return 'train' in self.mode

    @property
    def is_test_mode(self) -> bool:
        """
        """
        return self.mode == 'test'

    def padding_sample(self, samples: List[Sample]):
        maxlen = max([len(sample.labels) for sample in samples])
        padding_lens = [maxlen - len(sample.labels) for sample in samples]

        for padlen, sample in zip(padding_lens, samples):
            if padlen == 0:
                continue

            sample.labels = np.append(sample.labels,
                                      np.ones([padlen], dtype=np.int32) * -1)

            if sample.bboxes_2d is not None:
                empty_bbox = np.zeros([padlen, sample.bboxes_2d.shape[1]],
                                      np.float32)
                sample.bboxes_2d = BBoxes2D(
                    np.append(sample.bboxes_2d, empty_bbox, axis=0))

            if sample.bboxes_3d is not None:
                empty_bbox = np.zeros([padlen, sample.bboxes_3d.shape[1]],
                                      np.float32)
                sample.bboxes_3d = BBoxes3D(
                    np.append(sample.bboxes_3d, empty_bbox, axis=0))

    def collate_fn(self, batch: List):
        """
        """
        sample = batch[0]
        if isinstance(sample, np.ndarray):
            batch = np.stack(batch, axis=0)
            return batch
        elif isinstance(sample, paddle.Tensor):
            return layers.stack(batch, axis=0)
        elif isinstance(sample, numbers.Number):
            batch = np.array(batch)
            return batch
        elif isinstance(sample, (str, bytes)):
            return batch
        elif isinstance(sample, Sample):
            valid_keys = [
                key for key, value in sample.items() if value is not None
            ]
            self.padding_sample(batch)

            return {
                key: self.collate_fn([d[key] for d in batch])
                for key in valid_keys
            }
        elif isinstance(sample, Mapping):
            return {
                key: self.collate_fn([d[key] for d in batch])
                for key in sample
            }
        elif isinstance(sample, Sequence):
            sample_fields_num = len(sample)
            if not all(
                    len(sample) == sample_fields_num for sample in iter(batch)):
                raise RuntimeError(
                    "fileds number not same among samples in a batch")
            return [self.collate_fn(fields) for fields in zip(*batch)]

        raise TypeError(
            "batch data con only contains: tensor, numpy.ndarray, "
            "dict, list, number, paddle3d.Sample, but got {}".format(
                type(sample)))
