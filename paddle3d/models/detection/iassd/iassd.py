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

import os
from collections import defaultdict
from typing import List

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec

from paddle3d.apis import manager
from paddle3d.geometries import BBoxes3D, CoordMode
from paddle3d.models.layers import constant_init, reset_parameters
from paddle3d.ops import iou3d_nms_cuda, pointnet2_ops
from paddle3d.sample import Sample
from paddle3d.utils import box_utils
from paddle3d.utils.logger import logger

__all = ["IASSD"]


@manager.MODELS.add_component
class IASSD(nn.Layer):
    """Model of IA-SSD

    Args:
        backbone (nn.Layer): instantiated class of backbone.
        head (nn.Layer): instantiated class of head.
        post_process_cfg (dict): config of nms post-process.
    """

    def __init__(self, backbone, head, post_process_cfg):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.post_process_cfg = post_process_cfg
        self.export_model = False

        self.apply(self.init_weight)

        self.export_model = False

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                data: (B * N, C)  # C = [batch_id, x, y, z, intensity, ...]
                bboxes_3d: (B, num_gt, 8) # [x, y, z, l, w, h, heading, label]
        Returns:
            ...
        """

        batch_dict = self.backbone(batch_dict)
        batch_dict = self.head(batch_dict)

        if self.training:
            loss = self.head.get_loss()
            return {"loss": loss}
        else:
            if getattr(self, "export_model", False):
                return self.post_process(batch_dict)
            else:
                result_list = self.post_process(batch_dict)
                sample_list = self._parse_results_to_sample(
                    result_list, batch_dict)
                return {"preds": sample_list}

    def init_weight(self, m):
        if isinstance(m, nn.Conv2D):
            reset_parameters(m)
        elif isinstance(m, nn.Conv1D):
            reset_parameters(m)
        elif isinstance(m, nn.Linear):
            reset_parameters(m, reverse=True)
        elif isinstance(m, nn.BatchNorm2D):
            constant_init(m.weight, value=1)
            constant_init(m.bias, value=0)
        elif isinstance(m, nn.BatchNorm1D):
            constant_init(m.weight, value=1)
            constant_init(m.bias, value=0)

    def collate_fn(self, batch: List):
        data_dict = defaultdict(list)
        for cur_sample in batch:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch)

        collated_batch = {}
        collated_fileds = [
            "data", "bboxes_3d", "meta", "path", "modality", "calibs"
        ]

        for key, val in data_dict.items():
            if key not in collated_fileds or val[0] is None:
                continue
            if key == "data":
                collated_batch[key] = np.concatenate([
                    np.pad(
                        coor, ((0, 0), (1, 0)),
                        mode="constant",
                        constant_values=i) for i, coor in enumerate(val)
                ])

            elif key == "bboxes_3d":
                max_gt = max([len(x) for x in val])
                batch_bboxes_3d = np.zeros(
                    (batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                # pad num of bboxes to max_gt with zeros, as well as labels
                for k in range(batch_size):
                    batch_bboxes_3d[k, :val[k].__len__(), :] = val[k]
                collated_batch[key] = batch_bboxes_3d

            elif key in ["path", "modality", "calibs", "meta"]:
                collated_batch[key] = val

        collated_batch["batch_size"] = batch_size

        return collated_batch

    @paddle.no_grad()
    def post_process(self, batch_dict):
        batch_size = batch_dict["batch_size"]
        pred_list = []
        for index in range(batch_size):
            if batch_dict.get("batch_index", None) is not None:
                assert batch_dict["batch_box_preds"].shape.__len__() == 2
                batch_mask = batch_dict["batch_index"] == index
            else:
                assert batch_dict["batch_box_preds"].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict["batch_box_preds"][batch_mask]

            if not isinstance(batch_dict["batch_cls_preds"], list):
                cls_preds = batch_dict["batch_cls_preds"][batch_mask]
                if not batch_dict["cls_preds_normalized"]:
                    cls_preds = F.sigmoid(cls_preds)
            else:
                cls_preds = [
                    x[batch_mask] for x in batch_dict["batch_cls_preds"]
                ]
                if not batch_dict["cls_preds_normalized"]:
                    cls_preds = [F.sigmoid(x) for x in cls_preds]

            label_preds = paddle.argmax(cls_preds, axis=-1)
            cls_preds = paddle.max(cls_preds, axis=-1)
            selected_score, selected_label, selected_box = self.class_agnostic_nms(
                box_scores=cls_preds,
                box_preds=box_preds,
                label_preds=label_preds,
                nms_config=self.post_process_cfg["nms_config"],
                score_thresh=self.post_process_cfg["score_thresh"],
            )

            record_dict = {
                "pred_boxes": selected_box,
                "pred_scores": selected_score,
                "pred_labels": selected_label,
            }
            pred_list.append(record_dict)

        return pred_list

    def class_agnostic_nms(self, box_scores, box_preds, label_preds, nms_config,
                           score_thresh):
        scores_mask = paddle.nonzero(box_scores >= score_thresh)

        fake_score = paddle.to_tensor([0.0], dtype="float32")
        fake_label = paddle.to_tensor([-1.0], dtype="float32")
        fake_box = paddle.to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                                    dtype="float32")
        if paddle.shape(scores_mask)[0] == 0:
            return fake_score, fake_label, fake_box
        else:
            scores_mask = scores_mask
            box_scores = paddle.gather(box_scores, index=scores_mask)
            box_preds = paddle.gather(box_preds, index=scores_mask)
            label_preds = paddle.gather(label_preds, index=scores_mask)
            order = box_scores.argsort(0, descending=True)
            order = order[:nms_config["nms_pre_maxsize"]]
            box_preds = paddle.gather(box_preds, index=order)
            box_scores = paddle.gather(box_scores, index=order)
            label_preds = paddle.gather(label_preds, index=order)
            # When order is one-value tensor,
            # boxes[order] loses a dimension, so we add a reshape
            keep, num_out = iou3d_nms_cuda.nms_gpu(box_preds,
                                                   nms_config["nms_thresh"])
            if num_out.cast("int64") == 0:
                return fake_score, fake_label, fake_box
            else:
                selected = keep[0:num_out]
                selected = selected[:nms_config["nms_post_maxsize"]]
                selected_score = paddle.gather(box_scores, index=selected)
                selected_box = paddle.gather(box_preds, index=selected)
                selected_label = paddle.gather(label_preds, index=selected)
                return selected_score, selected_label, selected_box

    def _parse_results_to_sample(self, results, batch_dict):
        num = len(results)
        sample_list = []
        for i in range(num):
            result = results[i]
            path = batch_dict["path"][i]
            if (result["pred_labels"] == -1).any():
                sample = Sample(path=path, modality="lidar")
            else:
                sample = Sample(path=path, modality="lidar")
                box_preds = result["pred_boxes"]
                if isinstance(box_preds, paddle.Tensor):
                    box_preds = box_preds.numpy()
                # convert box format to kitti, only for kitti eval
                box_preds = box_utils.boxes3d_lidar_to_kitti_lidar(box_preds)
                cls_labels = result["pred_labels"]
                cls_scores = result["pred_scores"]
                sample.bboxes_3d = BBoxes3D(
                    box_preds,
                    origin=[0.5, 0.5, 0],
                    coordmode="Lidar",
                    rot_axis=2)
                sample.labels = cls_labels.numpy()
                sample.confidences = cls_scores.numpy()
                sample.alpha = (-np.arctan2(-box_preds[:, 1], box_preds[:, 0]) +
                                box_preds[:, 6])
                if ("calibs" in batch_dict) and (batch_dict["calibs"] is
                                                 not None):
                    sample.calibs = [
                        calib.numpy() for calib in batch_dict["calibs"][i]
                    ]
            if ("meta" in batch_dict) and (batch_dict["meta"] is not None):
                sample.meta.update(batch_dict["meta"][i])
            sample_list.append(sample)

        return sample_list

    def export(self, save_dir, **kwargs):
        self.export_model = True
        self.backbone.export_model = True
        input_spec = [{
            "data":
            InputSpec(shape=[-1, 4], name="data", dtype='float32')
        }]
        save_path = os.path.join(save_dir, 'iassd')

        paddle.jit.to_static(self, input_spec=input_spec)
        paddle.jit.save(self, save_path, input_spec=input_spec)

        logger.info("Exported model is saved in {}".format(save_path))
