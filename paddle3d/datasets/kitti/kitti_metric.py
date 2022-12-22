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

from typing import Dict, List

import numpy as np

from paddle3d.datasets.kitti.kitti_utils import (
    Calibration, box_lidar_to_camera, filter_fake_result)
from paddle3d.datasets.metrics import MetricABC
from paddle3d.geometries.bbox import (
    BBoxes2D, BBoxes3D, CoordMode, boxes3d_kitti_camera_to_imageboxes,
    boxes3d_lidar_to_kitti_camera, project_to_image)
from paddle3d.sample import Sample
from paddle3d.thirdparty import kitti_eval
from paddle3d.utils.logger import logger


class KittiMetric(MetricABC):
    """
    """

    def __init__(self, groundtruths: List[np.ndarray], classmap: Dict[int, str],
                 indexes: List):
        self.gt_annos = groundtruths
        self.predictions = []
        self.classmap = classmap
        self.indexes = indexes

    def _parse_gt_to_eval_format(self,
                                 groundtruths: List[np.ndarray]) -> List[dict]:
        res = []
        for rows in groundtruths:
            if rows.size == 0:
                res.append({
                    'name': np.zeros([0]),
                    'truncated': np.zeros([0]),
                    'occluded': np.zeros([0]),
                    'alpha': np.zeros([0]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.zeros([0]),
                    'score': np.zeros([0])
                })
            else:
                res.append({
                    'name': rows[:, 0],
                    'truncated': rows[:, 1].astype(np.float64),
                    'occluded': rows[:, 2].astype(np.int64),
                    'alpha': rows[:, 3].astype(np.float64),
                    'bbox': rows[:, 4:8].astype(np.float64),
                    'dimensions': rows[:, [10, 8, 9]].astype(np.float64),
                    'location': rows[:, 11:14].astype(np.float64),
                    'rotation_y': rows[:, 14].astype(np.float64)
                })

        return res

    def get_camera_box2d(self, bboxes_3d: BBoxes3D, proj_mat: np.ndarray):
        box_corners = bboxes_3d.corners_3d
        box_corners_in_image = project_to_image(box_corners, proj_mat)
        minxy = np.min(box_corners_in_image, axis=1)
        maxxy = np.max(box_corners_in_image, axis=1)
        box_2d_preds = BBoxes2D(np.concatenate([minxy, maxxy], axis=1))

        return box_2d_preds

    def _parse_predictions_to_eval_format(
            self, predictions: List[Sample]) -> List[dict]:
        res = {}
        for pred in predictions:
            filter_fake_result(pred)
            id = pred.meta.id
            if pred.bboxes_3d is None:
                det = {
                    'truncated': np.zeros([0]),
                    'occluded': np.zeros([0]),
                    'alpha': np.zeros([0]),
                    'name': np.zeros([0]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.zeros([0]),
                    'score': np.zeros([0]),
                }
            else:
                num_boxes = pred.bboxes_3d.shape[0]
                names = np.array(
                    [self.classmap[label] for label in pred.labels])
                calibs = pred.calibs

                alpha = pred.get('alpha', np.zeros([num_boxes]))

                if pred.bboxes_3d.coordmode != CoordMode.KittiCamera:
                    bboxes_3d = box_lidar_to_camera(pred.bboxes_3d, calibs)
                else:
                    bboxes_3d = pred.bboxes_3d

                if bboxes_3d.origin != [.5, 1., .5]:
                    bboxes_3d[:, :3] += bboxes_3d[:, 3:6] * (
                        np.array([.5, 1., .5]) - np.array(bboxes_3d.origin))
                    bboxes_3d.origin = [.5, 1., .5]

                if pred.bboxes_2d is None:
                    bboxes_2d = self.get_camera_box2d(bboxes_3d, calibs[2])
                else:
                    bboxes_2d = pred.bboxes_2d

                loc = bboxes_3d[:, :3]
                dim = bboxes_3d[:, 3:6]

                det = {
                    # fake value
                    'truncated': np.zeros([num_boxes]),
                    'occluded': np.zeros([num_boxes]),
                    # predict value
                    'alpha': alpha,
                    'name': names,
                    'bbox': bboxes_2d,
                    'dimensions': dim,
                    # TODO: coord trans
                    'location': loc,
                    'rotation_y': bboxes_3d[:, 6],
                    'score': pred.confidences,
                }

            res[id] = det

        return [res[idx] for idx in self.indexes]

    def update(self, predictions: List[Sample], **kwargs):
        """
        """
        self.predictions += predictions

    def compute(self, verbose=False, **kwargs) -> dict:
        """
        """
        gt_annos = self._parse_gt_to_eval_format(self.gt_annos)
        dt_annos = self._parse_predictions_to_eval_format(self.predictions)

        if len(dt_annos) != len(gt_annos):
            raise RuntimeError(
                'The number of predictions({}) is not equal to the number of GroundTruths({})'
                .format(len(dt_annos), len(gt_annos)))

        metric_r40_dict = kitti_eval(
            gt_annos,
            dt_annos,
            current_classes=list(self.classmap.values()),
            metric_types=["bbox", "bev", "3d"],
            recall_type='R40')

        metric_r11_dict = kitti_eval(
            gt_annos,
            dt_annos,
            current_classes=list(self.classmap.values()),
            metric_types=["bbox", "bev", "3d"],
            recall_type='R11')

        if verbose:
            for cls, cls_metrics in metric_r40_dict.items():
                logger.info("{}:".format(cls))
                for overlap_thresh, metrics in cls_metrics.items():
                    for metric_type, thresh in zip(["bbox", "bev", "3d"],
                                                   overlap_thresh):
                        if metric_type in metrics:
                            logger.info(
                                "{} AP_R40@{:.0%}: {:.2f} {:.2f} {:.2f}".format(
                                    metric_type.upper().ljust(4), thresh,
                                    *metrics[metric_type]))

            for cls, cls_metrics in metric_r11_dict.items():
                logger.info("{}:".format(cls))
                for overlap_thresh, metrics in cls_metrics.items():
                    for metric_type, thresh in zip(["bbox", "bev", "3d"],
                                                   overlap_thresh):
                        if metric_type in metrics:
                            logger.info(
                                "{} AP_R11@{:.0%}: {:.2f} {:.2f} {:.2f}".format(
                                    metric_type.upper().ljust(4), thresh,
                                    *metrics[metric_type]))
        return metric_r40_dict, metric_r11_dict


class KittiDepthMetric(MetricABC):
    """
    """

    def __init__(self, eval_gt_annos, class_names):
        self.eval_gt_annos = eval_gt_annos
        self.predictions = []
        self.class_names = class_names

    def generate_prediction_dicts(self,
                                  batch_dict,
                                  pred_dicts,
                                  output_path=None):
        """
        Args:
            batch_dict: list of batch_dict
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            output_path:
        Returns:
        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples),
                'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]),
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]),
                'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cast("int64").cpu().numpy()
            if pred_labels[0] < 0:
                pred_dict = get_template_prediction(0)
                return pred_dict

            pred_dict = get_template_prediction(pred_scores.shape[0])
            # calib = batch_dict['calib'][batch_index]
            calib = Calibration({
                "P2":
                batch_dict["trans_cam_to_img"][batch_index].cpu().numpy(),
                "R0":
                batch_dict["R0"][batch_index].cpu().numpy(),
                "Tr_velo2cam":
                batch_dict["Tr_velo2cam"][batch_index].cpu().numpy()
            })
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape)

            pred_dict['name'] = np.array(self.class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(
                -pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            annos.append(single_pred_dict)
        return annos

    def update(self, predictions, ground_truths, **kwargs):
        """
        """
        self.predictions += self.generate_prediction_dicts(
            ground_truths, predictions)

    def compute(self, verbose=False, **kwargs) -> dict:
        """
        """
        eval_gt_annos = self.eval_gt_annos
        eval_det_annos = self.predictions

        if len(eval_det_annos) != len(eval_gt_annos):
            raise RuntimeError(
                'The number of predictions({}) is not equal to the number of GroundTruths({})'
                .format(len(eval_det_annos), len(eval_gt_annos)))

        metric_dict = kitti_eval(eval_gt_annos, eval_det_annos,
                                 self.class_names)

        if verbose:
            for cls, cls_metrics in metric_dict.items():
                logger.info("{}:".format(cls))
                for overlap_thresh, metrics in cls_metrics.items():
                    overlap_thresh = overlap_thresh + overlap_thresh
                    for metric_type, thresh in zip(["bbox", "bev", "3d", "aos"],
                                                   overlap_thresh):
                        if metric_type in metrics:
                            logger.info(
                                "{} AP@{:.0%}: {:.2f} {:.2f} {:.2f}".format(
                                    metric_type.upper().ljust(4), thresh,
                                    *metrics[metric_type]))
        return metric_dict
