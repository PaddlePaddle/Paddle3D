# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import os
from typing import Dict, List

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec

from paddle3d.apis import manager
from paddle3d.geometries import BBoxes3D
from paddle3d.models.common.model_nms_utils import class_agnostic_nms
from paddle3d.sample import Sample, SampleMeta
from paddle3d.utils.logger import logger
from paddle3d.models.layers.param_init import uniform_init


@manager.MODELS.add_component
class PVRCNN(nn.Layer):
    def __init__(self, num_class, voxelizer, voxel_encoder, middle_encoder,
                 point_encoder, backbone, neck, dense_head, point_head,
                 roi_head, post_process_cfg):
        super(PVRCNN, self).__init__()
        self.num_class = num_class
        self.voxelizer = voxelizer
        self.voxel_encoder = voxel_encoder
        self.middle_encoder = middle_encoder
        self.point_encoder = point_encoder
        self.backbone = backbone
        self.neck = neck
        self.dense_head = dense_head
        self.point_head = point_head
        self.roi_head = roi_head
        self.post_process_cfg = post_process_cfg
        self.init_weights()

    def init_weights(self):
        need_uniform_init_bn_weight_modules = [
            self.middle_encoder, self.point_encoder.vsa_point_feature_fusion,
            self.backbone, self.neck, self.point_head,
            self.roi_head.shared_fc_layer, self.roi_head.cls_layers,
            self.roi_head.reg_layers
        ]
        for module in need_uniform_init_bn_weight_modules:
            for layer in module.sublayers():
                if 'BatchNorm' in layer.__class__.__name__:
                    uniform_init(layer.weight, 0, 1)

    def voxelize(self, points):
        voxels, coordinates, num_points_in_voxel = self.voxelizer(points)
        return voxels, coordinates, num_points_in_voxel

    def forward(self, batch_dict, **kwargs):
        voxel_features, coordinates, voxel_num_points = self.voxelizer(
            batch_dict['data'])
        batch_dict["voxel_coords"] = coordinates
        points_pad = []
        if not getattr(self, "in_export_mode", False):
            for bs_idx, point in enumerate(batch_dict['data']):
                point_dim = point.shape[-1]
                point = point.reshape([1, -1, point_dim])
                point_pad = F.pad(
                    point, [1, 0],
                    value=bs_idx,
                    mode='constant',
                    data_format="NCL")
                point_pad = point_pad.reshape([-1, point_dim + 1])
                points_pad.append(point_pad)
            batch_dict['points'] = paddle.concat(points_pad, axis=0)
        else:
            point = batch_dict['data']
            batch_dict['batch_size'] = 1
            point = point.unsqueeze(1)
            point_pad = F.pad(
                point, [1, 0], value=0, mode='constant', data_format="NCL")
            batch_dict['points'] = point_pad.squeeze(1)

        voxel_features = self.voxel_encoder(voxel_features, voxel_num_points)
        middle_out = self.middle_encoder(voxel_features,
                                         batch_dict['voxel_coords'],
                                         batch_dict['batch_size'])
        batch_dict.update(middle_out)
        backbone_out = self.backbone(middle_out['spatial_features'])
        batch_dict['spatial_features_2d'] = self.neck(backbone_out)
        batch_dict = self.dense_head(batch_dict)
        batch_dict = self.point_encoder(batch_dict)
        batch_dict = self.point_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)

        if self.training:
            return self.get_training_loss()
        else:
            pred_dicts = self.post_processing(batch_dict)
            if not getattr(self, "in_export_mode", False):
                preds = self._parse_results_to_sample(pred_dicts, batch_dict)
                return {'preds': preds}
            else:
                return pred_dicts[0]

    def collate_fn(self, batch: List):
        """
        """
        sample_merged = collections.defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                sample_merged[k].append(v)
        batch_size = len(sample_merged['meta'])
        ret = {}
        for key, elems in sample_merged.items():
            if key in ["meta"]:
                ret[key] = [elem.id for elem in elems]
            elif key in ["path", "modality", "calibs"]:
                ret[key] = elems
            elif key == "data":
                ret[key] = [elem for elem in elems]
            elif key in ['gt_boxes']:
                max_gt = max([len(x) for x in elems])
                batch_gt_boxes3d = np.zeros(
                    (batch_size, max_gt, elems[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :elems[k].__len__(), :] = elems[k]
                ret[key] = batch_gt_boxes3d
        ret['batch_size'] = batch_size

        return ret

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return {"loss": loss}

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = F.sigmoid(cls_preds)
            else:
                cls_preds = [
                    x[batch_mask] for x in batch_dict['batch_cls_preds']
                ]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [F.sigmoid(x) for x in cls_preds]

            if self.post_process_cfg["nms_config"]["multi_classes_nms"]:
                raise NotImplementedError
            else:
                label_preds = paddle.argmax(cls_preds, axis=-1)
                cls_preds = paddle.max(cls_preds, axis=-1)
                if self.dense_head.num_class > 1:
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                final_scores, final_labels, final_boxes = class_agnostic_nms(
                    box_scores=cls_preds,
                    box_preds=box_preds,
                    label_preds=label_preds,
                    nms_config=self.post_process_cfg["nms_config"],
                    score_thresh=self.post_process_cfg["score_thresh"])

            if not getattr(self, "in_export_mode", False):
                record_dict = {
                    'box3d_lidar': final_boxes,
                    'scores': final_scores,
                    'label_preds': final_labels
                }
                pred_dicts.append(record_dict)
            else:
                pred_dicts.append([final_boxes, final_scores, final_labels])

        return pred_dicts

    def _convert_origin_for_eval(self, sample: dict):
        if sample.bboxes_3d.origin != [.5, .5, 0]:
            sample.bboxes_3d[:, :3] += sample.bboxes_3d[:, 3:6] * (
                np.array([.5, .5, 0]) - np.array(sample.bboxes_3d.origin))
            sample.bboxes_3d.origin = [.5, .5, 0]
        return sample

    def _parse_results_to_sample(self, results: dict, sample: dict):
        num_samples = len(results)
        new_results = []
        for i in range(num_samples):
            data = Sample(sample["path"][i], sample["modality"][i])
            bboxes_3d = results[i]["box3d_lidar"].numpy()
            labels = results[i]["label_preds"].numpy() - 1
            confidences = results[i]["scores"].numpy()
            bboxes_3d[..., 3:5] = bboxes_3d[..., [4, 3]]
            bboxes_3d[..., -1] = -(bboxes_3d[..., -1] + np.pi / 2.)
            data.bboxes_3d = BBoxes3D(bboxes_3d)
            data.bboxes_3d.coordmode = 'Lidar'
            data.bboxes_3d.origin = [0.5, 0.5, 0.5]
            data.bboxes_3d.rot_axis = 2
            data.labels = labels
            data.confidences = confidences
            data.meta = SampleMeta(id=sample["meta"][i])
            if "calibs" in sample:
                data.calibs = [calib.numpy() for calib in sample["calibs"][i]]
            data = self._convert_origin_for_eval(data)
            new_results.append(data)
        return new_results

    def export(self, save_dir: str, **kwargs):
        self.in_export_mode = True
        self.voxelizer.in_export_mode = True
        self.middle_encoder.in_export_mode = True
        save_path = os.path.join(save_dir, 'pv_rcnn')
        points_shape = [-1, self.voxel_encoder.in_channels]

        input_spec = [{
            "data":
            InputSpec(shape=points_shape, name='data', dtype='float32')
        }]

        paddle.jit.to_static(self, input_spec=input_spec)
        paddle.jit.save(self, save_path, input_spec=input_spec)

        logger.info("Exported model is saved in {}".format(save_dir))
