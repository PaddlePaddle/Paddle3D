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

import tempfile
from paddle3d.datasets.metrics import MetricABC
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.data_classes import Box as NuScenesBox
import os.path as osp
import json
import pyquaternion
from nuscenes.eval.detection.config import config_factory
import pickle
import os
import numpy as np


class BevDetNuScenesMetric(MetricABC):
    def __init__(self,
                 data_root,
                 ann_file,
                 classes,
                 load_interval=1,
                 mode='val',
                 eval_version='detection_cvpr_2019',
                 modality=None,
                 ego_cam='CAM_FRONT'):
        super(BevDetNuScenesMetric, self).__init__()
        self.data_root = data_root
        self.eval_version = eval_version
        self.eval_detection_configs = config_factory(self.eval_version)
        self.result_path = None
        self.load_interval = load_interval
        self.CLASSES = classes

        self.DefaultAttribute = {
            'car': 'vehicle.parked',
            'pedestrian': 'pedestrian.moving',
            'trailer': 'vehicle.parked',
            'truck': 'vehicle.parked',
            'bus': 'vehicle.moving',
            'motorcycle': 'cycle.without_rider',
            'construction_vehicle': 'vehicle.parked',
            'bicycle': 'cycle.without_rider',
            'barrier': '',
            'traffic_cone': ''
        }
        self.modality = modality
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False)
        self.data_infos = self.load_annotations(ann_file)
        self.ego_cam = ego_cam
        self.predictions = []

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        """
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        """
        assert isinstance(results, list), 'results must be a list'
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = os.path.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = os.path.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(results):
            boxes = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']
            sample_token = self.data_infos[sample_id]['token']

            trans = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_translation']
            rot = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_rotation']
            rot = pyquaternion.Quaternion(rot)
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        os.makedirs(jsonfile_prefix, exist_ok=True)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        with open(res_path, 'w') as f:
            json.dump(nusc_submissions, f)
        return res_path

    def update(self, predictions, **kwargs):
        """
        """
        # format
        # [{'pts_bbox': {...}}, {'pts_bbox': {...}}]
        self.predictions += predictions

    def compute(self, **kwargs) -> dict:
        """Evaluation for a single model in nuScenes protocol.
        """

        # input self.predictions
        result_path_dict, tmp_dir = self.format_results(self.predictions)
        result_path = result_path_dict['pts_bbox']

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)

        nusc_eval.main(render_curves=False)
