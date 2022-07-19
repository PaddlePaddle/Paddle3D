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

from paddle3d.datasets.kitti.kitti_utils import box_lidar_to_camera
from paddle3d.datasets.metrics import MetricABC
from paddle3d.geometries.bbox import (BBoxes2D, BBoxes3D, CoordMode,
                                      project_to_image)
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
            id = pred.meta.id
            if pred.bboxes_2d is None and pred.bboxes_3d is None:
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

        metric_dict = kitti_eval(
            gt_annos,
            dt_annos,
            current_classes=list(self.classmap.values()),
            metric_types=["bbox", "bev", "3d"])

        if verbose:
            for cls, cls_metrics in metric_dict.items():
                logger.info("{}:".format(cls))
                for overlap_thresh, metrics in cls_metrics.items():
                    for metric_type, thresh in zip(["bbox", "bev", "3d"],
                                                   overlap_thresh):
                        if metric_type in metrics:
                            logger.info(
                                "{} AP@{:.0%}: {:.2f} {:.2f} {:.2f}".format(
                                    metric_type.upper().ljust(4), thresh,
                                    *metrics[metric_type]))
        return metric_dict
