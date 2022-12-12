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

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle3d.apis import manager
from paddle3d.geometries import BBoxes3D, CoordMode
from paddle3d.models.detection.pointpillars.pointpillars_coder import \
    PointPillarsCoder
from paddle3d.models.layers.layer_libs import rotate_nms_pcdet
from paddle3d.sample import Sample

__all__ = ["SSDHead"]


@manager.HEADS.add_component
class SSDHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 feature_channels=384,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 box_code_size=7,
                 nms_score_threshold=0.05,
                 nms_pre_max_size=1000,
                 nms_post_max_size=300,
                 nms_iou_threshold=0.5,
                 prediction_center_limit_range=None):
        super(SSDHead, self).__init__()

        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_direction_classifier = use_direction_classifier
        self.box_code_size = box_code_size
        self.nms_score_threshold = nms_score_threshold
        self.nms_pre_max_size = nms_pre_max_size
        self.nms_post_max_size = nms_post_max_size
        self.nms_iou_threshold = nms_iou_threshold

        if prediction_center_limit_range is not None:
            self._limit_pred = True
            self.pred_center_limit_range = paddle.to_tensor(
                prediction_center_limit_range)
        else:
            self._limit_pred = False

        if encode_background_as_zeros:
            self._num_classes = num_classes
        else:
            self._num_classes = num_classes + 1
        self.cls_head = nn.Conv2D(feature_channels,
                                  num_anchor_per_loc * self._num_classes, 1)
        self.box_head = nn.Conv2D(feature_channels,
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.dir_head = nn.Conv2D(feature_channels, num_anchor_per_loc * 2,
                                      1)

    def forward(self, features):
        batch_size = features.shape[0]
        cls_preds = self.cls_head(features).transpose((0, 2, 3, 1)).reshape(
            (batch_size, -1, self._num_classes))
        box_preds = self.box_head(features).transpose((0, 2, 3, 1)).reshape(
            (batch_size, -1, self.box_code_size))
        ret = dict(cls_preds=cls_preds, box_preds=box_preds)

        if self.use_direction_classifier:
            dir_preds = self.dir_head(features).transpose((0, 2, 3, 1)).reshape(
                (batch_size, -1, 2))
            ret.update(dict(dir_preds=dir_preds))

        return ret

    @paddle.no_grad()
    def post_process(self,
                     samples,
                     preds,
                     anchors,
                     anchors_mask,
                     batch_size=None):
        preds["box_preds"] = PointPillarsCoder.decode(preds["box_preds"],
                                                      anchors)
        if getattr(self, "in_export_mode", False):
            box_preds = preds["box_preds"].squeeze(0)
            cls_preds = preds["cls_preds"].squeeze(0)

            if self.use_direction_classifier:
                dir_preds = preds["dir_preds"].squeeze(0).reshape((-1, 2))

            return paddle.static.nn.cond(
                anchors_mask.any(),
                true_fn=lambda: self._single_post_process(
                    box_preds,
                    cls_preds,
                    dir_preds=dir_preds
                    if self.use_direction_classifier else None,
                    anchors_mask=anchors_mask),
                false_fn=lambda: self._box_empty())
        else:
            batch_box_preds = preds["box_preds"]
            batch_cls_preds = preds["cls_preds"]

            if self.use_direction_classifier:
                batch_dir_preds = preds["dir_preds"].reshape((batch_size, -1,
                                                              2))

            results = []
            for i in range(batch_size):
                result = paddle.static.nn.cond(
                    anchors_mask[i].any(),
                    true_fn=lambda: self._single_post_process(
                        batch_box_preds[i],
                        batch_cls_preds[i],
                        dir_preds=batch_dir_preds[i]
                        if self.use_direction_classifier else None,
                        anchors_mask=anchors_mask[i]),
                    false_fn=lambda: self._box_empty())
                results.append(
                    self._parse_result_to_sample(
                        result, samples["path"][i], samples["calibs"][i], {
                            key: value[i]
                            for key, value in samples["meta"].items()
                        }))
            return {"preds": results}

    @paddle.no_grad()
    def _single_post_process(self,
                             box_preds,
                             cls_preds,
                             dir_preds=None,
                             anchors_mask=None):
        box_preds = box_preds[anchors_mask]
        cls_preds = cls_preds[anchors_mask]

        if self.encode_background_as_zeros:
            cls_confs = F.sigmoid(cls_preds)
        else:
            cls_confs = F.sigmoid(cls_preds[..., 1:])
        cls_scores = cls_confs.max(-1)
        cls_labels = cls_confs.argmax(-1)

        if self.use_direction_classifier:
            dir_preds = dir_preds[anchors_mask]
            dir_labels = dir_preds.argmax(axis=-1)

        kept = cls_scores >= self.nms_score_threshold
        if self._limit_pred:
            distance_kept = (box_preds[..., :3] >= self.pred_center_limit_range[:3]).all(1) \
                            & (box_preds[..., :3] <= self.pred_center_limit_range[3:]).all(1)
            kept = kept & distance_kept

        return paddle.static.nn.cond(
            kept.any(),
            true_fn=lambda: self._box_not_empty(
                box_preds[kept],
                cls_scores[kept],
                cls_labels[kept],
                dir_labels=dir_labels[kept]
                if self.use_direction_classifier else None),
            false_fn=lambda: self._box_empty())

    def _box_empty(self):
        pretiction_dict = {
            'box3d_lidar': paddle.zeros([1, self.box_code_size],
                                        dtype="float32"),
            'scores': -paddle.ones([1], dtype="float32"),
            'label_preds': -paddle.ones([1], dtype="int64")
        }
        return pretiction_dict

    def _box_not_empty(self, box_preds, cls_scores, cls_labels, dir_labels):
        if self.use_direction_classifier:
            box_preds[..., 6] += paddle.where(
                (box_preds[..., 6] > 0) ^ dir_labels.astype("bool"),
                paddle.to_tensor(math.pi), paddle.to_tensor(0.))
        # bottom center to object center
        box_preds[:, 2] = box_preds[:, 2] + box_preds[:, 5] * 0.5

        selected = rotate_nms_pcdet(
            box_preds,
            cls_scores,
            pre_max_size=self.nms_pre_max_size,
            post_max_size=self.nms_post_max_size,
            thresh=self.nms_iou_threshold)

        box_preds = paddle.index_select(box_preds, selected, axis=0)
        # object center to bottom center
        box_preds[:, 2] = box_preds[:, 2] - box_preds[:, 5] * 0.5
        cls_labels = paddle.index_select(cls_labels, selected, axis=0)
        cls_scores = paddle.index_select(cls_scores, selected, axis=0)

        prediction_dict = {
            'box3d_lidar': box_preds,
            'scores': cls_scores,
            'label_preds': cls_labels
        }
        return prediction_dict

    @staticmethod
    def _parse_result_to_sample(result, path, calibs, meta):
        if (result["scores"] == -1).any():
            sample = Sample(path=path, modality="lidar")
        else:
            sample = Sample(path=path, modality="lidar")
            sample.calibs = [calib.numpy() for calib in calibs]
            box_preds = result["box3d_lidar"]
            cls_labels = result["label_preds"]
            cls_scores = result["scores"]
            sample.bboxes_3d = BBoxes3D(
                box_preds.numpy(),
                origin=[.5, .5, 0],
                coordmode=CoordMode.KittiLidar,
                rot_axis=2)
            sample.labels = cls_labels.numpy()
            sample.confidences = cls_scores.numpy()
            sample.alpha = (-paddle.atan2(-box_preds[:, 1], box_preds[:, 0]) +
                            box_preds[:, 6]).numpy()

        sample.meta.update(meta)

        return sample
