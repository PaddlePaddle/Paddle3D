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
    def __init__(self, nuscense: NuScenes, mode: str, channel: str,
                 class_names: list, attrmap: dict):
        self.nusc = nuscense
        self.mode = mode
        self.channel = channel
        self.class_names = class_names
        self.attrmap = attrmap
        self.predictions = []

    def _parse_predictions_to_eval_format(self, predictions: List[Sample],
                                          eval_config) -> dict:
        # Nuscenes eval format:
        # https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any
        res = {}
        index = 0
        for pred in predictions:
            filter_fake_result(pred)
            num_boxes = pred.bboxes_3d.shape[0]

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
            # print('aaaa', pred.meta.id)
            # print('bbbb', channel_pose["rotation"])
            # print('cccc', channel_pose["translation"])
            # print('dddd', ego_pose["rotation"])
            # print('eeee', ego_pose["translation"])
            # import sys
            # sys.exit(0)

            global_box_list = []
            for box in nus_box_list:
                # Move box to ego vehicle coord system
                box.rotate(Quaternion(channel_pose["rotation"]))
                box.translate(np.array(channel_pose["translation"]))
                # filter det in ego.
                # cls_range_map = eval_config.class_range
                # radius = np.linalg.norm(box.center[:2], 2)
                # det_range = cls_range_map[self.class_names[box.label]]
                # if radius > det_range:
                #     continue
                # Move box to global coord system
                box.rotate(Quaternion(ego_pose["rotation"]))
                box.translate(np.array(ego_pose["translation"]))
                global_box_list.append(box)

            res[pred.meta.id] = []
            # for idx in range(num_boxes):
            for idx in range(len(global_box_list)):
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
            # 'meta': {
            #     'use_camera': True,
            #     'use_lidar': False,
            #     'use_radar': False,
            #     'use_map': False,
            #     'use_external': True,
            # },
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
            self._parse_predictions_to_eval_format(self.predictions,
                                                   eval_config))

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

            metrics_summary = evaluator.main(plot_examples=0,
                                             render_curves=False)
            metric_file = os.path.join(tmpdir, 'metrics_summary.json')
            with open(metric_file, 'r') as file:
                metrics = json.load(file)

        return metrics
