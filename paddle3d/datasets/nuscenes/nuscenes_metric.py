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
import os.path as osp
from typing import List

import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.data_classes import Box as NuScenesBox

import paddle
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


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


class NuScenesSegMetric(MetricABC):
    """
    """
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    def __init__(self,
                 class_names: list,
                 data_infos: list,
                 modality: dict,
                 version: str,
                 dataset_root: str,
                 result_names: list = ['pts_bbox']):
        self.class_names = class_names
        self.data_infos = data_infos
        self.modality = modality
        self.result_names = result_names
        self.version = version
        self.dataset_root = dataset_root
        self.predictions = []

    def _format_bbox(self, results, tmpdir):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            tmpdir (str): The prefix of the output jsonfile.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.class_names

        print('Start to convert detection format...')
        for sample_id, det in enumerate(results):
            annos = []
            boxes = output_to_nusc_box(det)

            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id], boxes, mapped_class_names,
                self.eval_detection_configs, self.eval_version)

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
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
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mkdir_or_exist(tmpdir)
        res_path = os.path.join(tmpdir, 'results_nusc.json')
        print('Results writes to', res_path)
        with open(res_path, 'w') as file:
            json.dump(nusc_submissions, file, cls=StructureEncoder)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         ret_iou,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """

        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval
        ret_ious = []
        for i in ret_iou:
            ret_ious.append(i.item())
        print(ret_ious)
        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.dataset_root, verbose=False)
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

        # record metrics
        with open(osp.join(output_dir, 'metrics_summary.json'), 'r') as file:
            metrics = json.load(file)
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.class_names:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        detail['iou'] = ret_ious

        return detail

    def format_results(self, results, result_names=['pts_bbox']):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self.data_infos), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self.data_infos)))

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        with generate_tempdir() as tmpdir:
            if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
                result_files = self._format_bbox(results, tmpdir)
            else:
                # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
                result_files = dict()
                for name in ['pts_bbox']:
                    print(f'\nFormating bboxes of {name}')
                    results_ = [out[name] for out in results]
                    tmp_file_ = osp.join(tmpdir, name)
                    result_files.update(
                        {name: self._format_bbox(results_, tmp_file_)})
            res = [0, 0, 0]
            for i in range(len(self.predictions)):
                for tt in range(3):
                    res[tt] += self.predictions[i]['ret_iou'][tt]
            n = len(self.predictions)
            for i in range(len(res)):
                res[i] = res[i] / n

            if isinstance(result_files, dict):
                results_dict = dict()
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    ret_dict = self._evaluate_single(result_files[name], res)
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(result_files)
        return results_dict

    def update(self, predictions: List[Sample], **kwargs):
        """
        """
        self.predictions.extend(predictions)

    def compute(self, **kwargs) -> dict:
        """
        """
        self.eval_version = 'detection_cvpr_2019'
        self.eval_detection_configs = config_factory(self.eval_version)
        results_dict = self.format_results(self.predictions)
        return results_dict


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    def get_gravity_center(bboxes):
        bottom_center = bboxes[:, :3]
        gravity_center = paddle.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + bboxes[:, 5] * 0.5
        return gravity_center

    box_gravity_center = get_gravity_center(box3d).numpy()
    box_dims = box3d[:, 3:6].numpy()
    box_yaw = box3d[:, 6].numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d[i, 7:9].numpy(), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
