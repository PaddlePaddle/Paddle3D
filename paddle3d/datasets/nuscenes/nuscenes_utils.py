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

import operator
from typing import Tuple

import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from paddle3d.geometries.bbox import BBoxes2D, BBoxes3D
from paddle3d.sample import Sample

# cls_attr_dist refers to https://github.com/tianweiy/CenterPoint/blob/master/det3d/datasets/nuscenes/nusc_common.py#L47
cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}


def filter_fake_result(detection: Sample):
    box3d = detection.bboxes_3d
    velocities = detection.bboxes_3d.velocities
    scores = detection.confidences
    labels = detection.labels
    box_list = []
    velocity_list = []
    score_list = []
    label_list = []
    for i in range(box3d.shape[0]):
        if scores[i] < 0:
            continue
        box_list.append(box3d[i])
        velocity_list.append(velocities[i])
        score_list.append(scores[i])
        label_list.append(labels[i])
    detection.bboxes_3d = BBoxes3D(
        np.asarray(box_list), velocities=np.asarray(velocity_list))
    detection.labels = np.asarray(label_list)
    detection.confidences = np.asarray(score_list)


def second_bbox_to_nuscenes_box(pred_sample: Sample):
    """
    This function refers to https://github.com/tianweiy/CenterPoint/blob/master/det3d/datasets/nuscenes/nusc_common.py#L160
    """
    pred_sample.bboxes_3d[:, -1] = -pred_sample.bboxes_3d[:, -1] - np.pi / 2
    nuscenes_box_list = []
    for i in range(pred_sample.bboxes_3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=pred_sample.bboxes_3d[i, -1])
        velocity = (*pred_sample.bboxes_3d.velocities[i, 0:2], 0.0)
        box = Box(
            pred_sample.bboxes_3d[i, :3],
            pred_sample.bboxes_3d[i, 3:6],
            quat,
            label=pred_sample.labels[i],
            score=pred_sample.confidences[i],
            velocity=velocity,
        )
        nuscenes_box_list.append(box)
    return nuscenes_box_list


def get_nuscenes_box_attribute(box: Box, label_name: str):
    if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
        if label_name in [
                "car",
                "construction_vehicle",
                "bus",
                "truck",
                "trailer",
        ]:
            attr = "vehicle.moving"
        elif label_name in ["bicycle", "motorcycle"]:
            attr = "cycle.with_rider"
        else:
            attr = None
    else:
        if label_name in ["pedestrian"]:
            attr = "pedestrian.standing"
        elif label_name in ["bus"]:
            attr = "vehicle.stopped"
        else:
            attr = None

    if attr is None:
        attr = max(
            cls_attr_dist[label_name].items(), key=operator.itemgetter(1))[0]
    return attr
