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

import abc
import contextlib
import os
from typing import List

import paddle
import paddle.nn as nn

from paddle3d.geometries import CoordMode


class BaseDetectionModel(abc.ABC, nn.Layer):
    def __init__(self, box_with_velocity: bool = False):
        super().__init__()
        self.box_with_velocity = box_with_velocity
        self.in_export_mode = False

    @property
    def input_spec(self) -> paddle.static.InputSpec:
        """
        """
        return [[paddle.static.InputSpec(**_input) for _input in self.inputs]]

    @abc.abstractproperty
    def inputs(self) -> List[dict]:
        """
        """

    def outputs(self) -> List[dict]:
        """
        """
        boxdim = 7 if not self.box_with_velocity else 9
        box3ds = {'name': 'box3d', 'dtype': 'float32', 'shape': [None, boxdim]}
        labels = {'name': 'label', 'dtype': 'int32', 'shape': [None]}
        confidences = {
            'name': 'confidence',
            'dtype': 'float32',
            'shape': [None]
        }
        return [box3ds, labels, confidences]

    def forward(self, samples):
        if self.in_export_mode:
            return self.export_forward(samples)
        elif self.training:
            return self.train_forward(samples)

        return self.test_forward(samples)

    @abc.abstractproperty
    def coord(self) -> CoordMode:
        """
        """

    @abc.abstractproperty
    def sensor(self) -> str:
        """
        """

    def set_export_mode(self, mode: bool = True):
        for sublayer in self.sublayers(include_self=True):
            sublayer.in_export_mode = mode

    # @abc.abstractmethod
    # def preprocess(self):
    #     """
    #     """

    # @abc.abstractmethod
    # def test_forward(self):
    #     """
    #     """

    # @abc.abstractmethod
    # def train_forward(self):
    #     """
    #     """

    # @abc.abstractmethod
    # def export_forward(self):
    #     """
    #     """

    @contextlib.contextmanager
    def exporting(self):
        self.set_export_mode(True)
        yield
        self.set_export_mode(False)

    def export(self, save_dir: str, with_preprocess: bool = False):
        with self.exporting():
            paddle.jit.to_static(self, input_spec=self.input_spec)
            paddle.jit.save(self, os.path.join(save_dir, "inference"))

    # def _parse_results_to_sample(self, results: dict, sample: dict):
    #     num_samples = len(results)
    #     new_results = []
    #     for i in range(num_samples):
    #         data = Sample(sample["path"][i], sample["modality"][i])
    #         bboxes_3d = results[i]["box3d_lidar"].numpy()
    #         labels = results[i]["label_preds"].numpy()
    #         confidences = results[i]["scores"].numpy()
    #         data.bboxes_3d = BBoxes3D(bboxes_3d[:, [0, 1, 2, 3, 4, 5, -1]])
    #         data.bboxes_3d.coordmode = 'Lidar'
    #         data.bboxes_3d.origin = [0.5, 0.5, 0.5]
    #         data.bboxes_3d.rot_axis = 2
    #         if bboxes_3d.shape[-1] == 9:
    #             data.bboxes_3d.velocities = bboxes_3d[:, 6:8]
    #         data.labels = labels
    #         data.confidences = confidences
    #         data.meta = SampleMeta(id=results[i]["meta"])
    #         if "calibs" in sample:
    #             calib = [calibs.numpy()[i] for calibs in sample["calibs"]]
    #             data.calibs = calib
    #         new_results.append(data)
    #     return new_results
