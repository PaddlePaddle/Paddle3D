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

from typing import List

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

from paddle3d.sample import Sample
from paddle3d.utils import box_utils
from paddle3d.utils.logger import logger

tf.get_logger().setLevel('INFO')


class WaymoMetric(tf.test.TestCase):
    """
    AP/APH metric evaluation of Waymo Dataset.
    This code is beased on:
        <https://github.com/yifanzhang713/IA-SSD/blob/main/pcdet/datasets/waymo/waymo_eval.py>
    """

    WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']

    def __init__(self, gt_infos, class_names, distance_thresh):
        self.gt_infos = gt_infos
        self.class_names = class_names
        self.distance_thresh = distance_thresh
        self.predictions = []

    def generate_prediction_infos(self, sample_list: List[Sample]):
        prediction_infos = []
        for sample in sample_list:
            # there is no det
            if sample["labels"] is None:
                pred_dict = {
                    'name': np.zeros(0),
                    'score': np.zeros(0),
                    'boxes_lidar': np.zeros([0, 7])
                }
            # there is a det
            else:
                pred_dict = {
                    "name": np.array(self.class_names)[sample["labels"]],
                    "score": np.array(sample["confidences"]),
                    "boxes_lidar": np.array(sample["bboxes_3d"])
                }
            prediction_infos.append(pred_dict)
        return prediction_infos

    def parse_infos_to_eval_format(self,
                                   infos: List[dict],
                                   class_names,
                                   is_gt=True,
                                   is_kitti=True):
        frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty = [], [], [], [], [], []

        for frame_index, info in enumerate(infos):
            if is_gt:
                box_mask = np.array([n in class_names for n in info['name']],
                                    dtype=np.bool_)
                if 'num_points_in_gt' in info:
                    zero_difficulty_mask = info['difficulty'] == 0
                    info['difficulty'][(info['num_points_in_gt'] > 5)
                                       & zero_difficulty_mask] = 1
                    info['difficulty'][(info['num_points_in_gt'] <= 5)
                                       & zero_difficulty_mask] = 2
                    nonzero_mask = info['num_points_in_gt'] > 0
                    box_mask = box_mask & nonzero_mask
                else:
                    logger.info(
                        'Please provide the num_points_in_gt for evaluating on Waymo Dataset'
                    )
                    raise NotImplementedError

                num_boxes = box_mask.sum()
                box_name = info['name'][box_mask]

                difficulty.append(info['difficulty'][box_mask])
                score.append(np.ones(num_boxes))
                boxes3d.append(info['gt_boxes_lidar'][box_mask])

            else:
                num_boxes = len(info['boxes_lidar'])
                difficulty.append([0] * num_boxes)
                score.append(info['score'])
                if is_kitti:
                    info[
                        'boxes_lidar'] = box_utils.boxes3d_kitti_lidar_to_lidar(
                            info['boxes_lidar'])
                boxes3d.append(np.array(info['boxes_lidar']))
                box_name = info['name']

            obj_type += [
                self.WAYMO_CLASSES.index(name)
                for i, name in enumerate(box_name)
            ]
            frame_id.append(np.array([frame_index] * num_boxes))
            overlap_nlz.append(np.zeros(num_boxes))  # set zero currently

        frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
        boxes3d = np.concatenate(boxes3d, axis=0)
        obj_type = np.array(obj_type).reshape(-1)
        score = np.concatenate(score).reshape(-1)
        overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
        difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

        boxes3d[:, -1] = box_utils.limit_period(
            boxes3d[:, -1], offset=0.5, period=np.pi * 2)

        return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty

    def mask_by_distance(self, distance_thresh, boxes_3d, *args):
        mask = np.linalg.norm(boxes_3d[:, 0:2], axis=1) < distance_thresh + 0.5
        boxes_3d = boxes_3d[mask]
        ret_ans = [boxes_3d]
        for arg in args:
            ret_ans.append(arg[mask])

        return tuple(ret_ans)

    def build_config(self):
        config = metrics_pb2.Config()
        config_text = """
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        levels:1
        levels:2
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.0
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_3D
        """

        for x in range(0, 100):
            config.score_cutoffs.append(x * 0.01)
        config.score_cutoffs.append(1.0)

        text_format.Merge(config_text, config)
        return config

    def build_graph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_overlap_nlz = tf.compat.v1.placeholder(dtype=tf.bool)

            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self.build_config(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=self._pd_overlap_nlz,
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=self._gt_difficulty,
            )
            return metrics

    def run_eval_ops(
            self,
            sess,
            graph,
            metrics,
            prediction_frame_id,
            prediction_bbox,
            prediction_type,
            prediction_score,
            prediction_overlap_nlz,
            ground_truth_frame_id,
            ground_truth_bbox,
            ground_truth_type,
            ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._pd_overlap_nlz: prediction_overlap_nlz,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            },
        )

    def eval_value_ops(self, sess, graph, metrics):
        return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

    def update(self,
               predictions: List[Sample],
               ground_truths: List[Sample] = None):
        self.predictions += predictions

    def compute(self, verbose):
        prediction_infos = self.generate_prediction_infos(self.predictions)
        assert len(prediction_infos) == len(self.gt_infos)

        tf.compat.v1.disable_eager_execution()
        # set is_kitti=True, because iassd's outputs is in kitti format
        pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz, _ = self.parse_infos_to_eval_format(
            prediction_infos, self.class_names, is_gt=False, is_kitti=True)
        gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty = self.parse_infos_to_eval_format(
            self.gt_infos, self.class_names, is_gt=True, is_kitti=False)

        pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz = self.mask_by_distance(
            self.distance_thresh, pd_boxes3d, pd_frameid, pd_type, pd_score,
            pd_overlap_nlz)
        gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty = self.mask_by_distance(
            self.distance_thresh, gt_boxes3d, gt_frameid, gt_type, gt_score,
            gt_difficulty)

        logger.info('Number: (pd, %d) VS. (gt, %d)' % (len(pd_boxes3d),
                                                       len(gt_boxes3d)))
        logger.info('Level 1: %d, Level2: %d)' % ((gt_difficulty == 1).sum(),
                                                  (gt_difficulty == 2).sum()))

        graph = tf.Graph()
        metrics = self.build_graph(graph)
        with self.session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            self.run_eval_ops(
                sess,
                graph,
                metrics,
                pd_frameid,
                pd_boxes3d,
                pd_type,
                pd_score,
                pd_overlap_nlz,
                gt_frameid,
                gt_boxes3d,
                gt_type,
                gt_difficulty,
            )
            with tf.compat.v1.variable_scope('detection_metrics', reuse=True):
                aps = self.eval_value_ops(sess, graph, metrics)

        if verbose:
            for k, v in aps.items():
                logger.info("{}: {:.4f}".format(k, v[0]))

        return aps
