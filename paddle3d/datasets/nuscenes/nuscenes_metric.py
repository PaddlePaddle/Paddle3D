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

import json
import os
from typing import List

import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

from paddle3d.datasets.metrics import MetricABC
from paddle3d.geometries import StructureEncoder
from paddle3d.sample import Sample
from paddle3d.utils.common import generate_tempdir

from .nuscenes_utils import (filter_fake_result, get_nuscenes_box_attribute,
                             second_bbox_to_nuscenes_box)


class NuScenesMetric(MetricABC):
    """
    """

    def __init__(self,
                 nuscense: NuScenes,
                 mode: str,
                 channel: str,
                 class_names: list,
                 attrmap: dict,
                 eval_version='detection_cvpr_2019'):
        self.nusc = nuscense
        self.mode = mode
        self.channel = channel
        self.class_names = class_names
        self.attrmap = attrmap
        self.predictions = []
        from nuscenes.eval.detection.config import config_factory
        self.eval_version = eval_version
        self.eval_detection_configs = config_factory(self.eval_version)

    def _parse_predictions_to_eval_format(self,
                                          predictions: List[Sample]) -> dict:
        # Nuscenes eval format:
        # https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any
        res = {}
        for pred in predictions:
            filter_fake_result(pred)

            # transform bboxes from second format to nuscenes format
            nus_box_list = second_bbox_to_nuscenes_box(pred)

            # from sensor pose to global pose
            sample = self.nusc.get('sample', pred.meta.id)
            sample_data = self.nusc.get('sample_data',
                                        sample['data'][self.channel])
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            channel_pose = self.nusc.get('calibrated_sensor',
                                         sample_data['calibrated_sensor_token'])
            ego_quaternion = Quaternion(ego_pose['rotation'])
            channel_quaternion = Quaternion(channel_pose['rotation'])
            global_box_list = []
            for box in nus_box_list:
                # Move box to ego vehicle coord system
                box.rotate(Quaternion(channel_pose["rotation"]))
                box.translate(np.array(channel_pose["translation"]))

                # filter det in ego.
                # TODO(luoqianhui): where this filter is need?
                cls_range_map = self.eval_detection_configs.class_range
                radius = np.linalg.norm(box.center[:2], 2)
                det_range = cls_range_map[self.class_names[box.label]]
                if radius > det_range:
                    continue

                # Move box to global coord system
                box.rotate(Quaternion(ego_pose["rotation"]))
                box.translate(np.array(ego_pose["translation"]))
                global_box_list.append(box)

            num_boxes = len(global_box_list)
            res[pred.meta.id] = []
            for idx in range(num_boxes):
                box = global_box_list[idx]
                label_name = self.class_names[box.label]
                attr = get_nuscenes_box_attribute(box, label_name)

                res[pred.meta.id].append({
                    'sample_token':
                    pred.meta.id,
                    'translation':
                    box.center.tolist(),
                    'size':
                    box.wlh.tolist(),
                    'rotation':
                    box.orientation.elements.tolist(),
                    'detection_name':
                    label_name,
                    'detection_score':
                    box.score,
                    'velocity':
                    box.velocity[:2].tolist(),
                    'attribute_name':
                    attr
                })

        return res

    def update(self, predictions: List[Sample], **kwargs):
        """
        """
        self.predictions += predictions

    def compute(self, **kwargs) -> dict:
        """
        """
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_version = 'detection_cvpr_2019'
        eval_config = config_factory(eval_version)

        dt_annos = {
            'meta': {
                'use_camera': True if self.channel.startswith('CAM') else False,
                'use_lidar': True if self.channel == 'LIDAR_TOP' else False,
                'use_radar': False,
                'use_map': False,
                'use_external': False,
            },
            'results': {}
        }
        dt_annos['results'].update(
            self._parse_predictions_to_eval_format(self.predictions))

        with generate_tempdir() as tmpdir:
            result_file = os.path.join(tmpdir, 'nuscenes_pred.json')
            with open(result_file, 'w') as file:
                json.dump(dt_annos, file, cls=StructureEncoder)

            evaluator = NuScenesEval(
                self.nusc,
                config=eval_config,
                result_path=result_file,
                eval_set=self.mode,
                output_dir=tmpdir,
                verbose=False,
            )

            metrics_summary = evaluator.main(
                plot_examples=0, render_curves=False)
            metric_file = os.path.join(tmpdir, 'metrics_summary.json')
            with open(metric_file, 'r') as file:
                metrics = json.load(file)

        return metrics
