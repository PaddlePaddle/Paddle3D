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
import numpy as np
from pyquaternion import Quaternion

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec

from paddle3d.apis import manager
from paddle3d.utils import checkpoint
from paddle3d.utils.logger import logger
from paddle3d.models.losses import unproject_points2d
from paddle3d.geometries import BBoxes3D, CoordMode
from paddle3d.sample import Sample, SampleMeta


@manager.MODELS.add_component
class DD3D(nn.Layer):
    """
    """

    def __init__(self,
                 backbone,
                 feature_locations_offset,
                 fpn,
                 fcos2d_head,
                 fcos2d_loss,
                 fcos2d_inference,
                 fcos3d_head,
                 fcos3d_loss,
                 fcos3d_inference,
                 prepare_targets,
                 do_nms,
                 nusc_sample_aggregate,
                 num_classes,
                 pixel_mean,
                 pixel_std,
                 input_strides,
                 size_divisibility,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.feature_locations_offset = feature_locations_offset
        self.fpn = fpn
        self.fcos2d_head = fcos2d_head
        self.fcos2d_loss = fcos2d_loss
        self.fcos2d_inference = fcos2d_inference
        self.only_box2d = True
        self.fcos3d_head = fcos3d_head
        if self.fcos3d_head is not None:
            self.only_box2d = False
        self.fcos3d_loss = fcos3d_loss
        self.fcos3d_inference = fcos3d_inference
        self.prepare_targets = prepare_targets
        self.do_nms = do_nms
        # nuScenes inference aggregates detections over all 6 cameras.
        self.nusc_sample_aggregate_in_inference = nusc_sample_aggregate
        self.num_classes = num_classes
        self.register_buffer(
            "pixel_mean",
            paddle.to_tensor(pixel_mean).reshape([1, -1, 1, 1]))
        self.register_buffer("pixel_std",
                             paddle.to_tensor(pixel_std).reshape([1, -1, 1, 1]))
        self.input_strides = input_strides
        self.size_divisibility = size_divisibility
        self.pretrained = pretrained
        self.init_weight()

    def preprocess_image(self, x, size_divisibility):
        x = (x.cast('float32') - self.pixel_mean) / self.pixel_std
        h_old, w_old = x.shape[-2:]
        h_new = (
            (h_old +
             (size_divisibility - 1)) // size_divisibility) * size_divisibility
        w_new = (
            (w_old +
             (size_divisibility - 1)) // size_divisibility) * size_divisibility
        x = F.pad(
            x, [0, w_new - w_old, 0, h_new - h_old], value=0.0, mode='constant')
        return x

    def preprocess_box3d(self, box3d, intrinsic):
        box_pose = box3d[:, :, 0:4]
        tvec = box3d[:, :, 4:7]
        proj_ctr = paddle.mm(
            intrinsic.unsqueeze(1).tile([1, tvec.shape[1], 1, 1]),
            tvec.unsqueeze(-1)).squeeze(-1)
        proj_ctr = proj_ctr[:, :, :2] / proj_ctr[:, :, 2:]
        box3d_new = paddle.concat([box3d[:, :, 0:4], proj_ctr, box3d[:, :, 6:]],
                                  axis=2)
        return box3d_new

    def forward(self, samples):
        images = self.preprocess_image(samples["data"], self.size_divisibility)
        samples["bboxes_3d"] = self.preprocess_box3d(
            samples["bboxes_3d"], samples['meta']['camera_intrinsic'])
        features = self.backbone(images)
        features = self.fpn(features)
        features = [features[f] for f in features.keys()]
        locations = self.compute_locations(features)
        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth = self.fcos3d_head(
                features)
        inv_intrinsics = samples['meta']['camera_intrinsic'].inverse()

        if self.training:
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(
                locations, samples["bboxes_2d"], samples["bboxes_3d"],
                samples["labels"], feature_shapes)

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(
                logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf,
                    dense_depth, inv_intrinsics, fcos2d_info, training_targets)
                losses.update(fcos3d_loss)
            loss_total = 0
            for key in losses.keys():
                loss_total += losses[key]
            losses.update({'loss': loss_total})
            return losses
        else:
            pred_instances, fcos2d_info = self.fcos2d_inference(
                logits, box2d_reg, centerness, locations)
            if not self.only_box2d:
                self.fcos3d_inference(box3d_quat, box3d_ctr, box3d_depth,
                                      box3d_size, box3d_conf, inv_intrinsics,
                                      pred_instances, fcos2d_info)
                score_key = "scores_3d"
            else:
                score_key = "scores"

            # Transpose to "image-first", i.e. (B, L)
            pred_instances = list(zip(*pred_instances))
            pred_instances_cat = []
            for i, pred_instance in enumerate(pred_instances):
                pred_instance_cat = {}
                for key in pred_instance[0].keys():
                    sum_c = sum([
                        pred_instance[j][key].shape[0]
                        for j in range(len(pred_instance))
                    ])
                    if sum_c == 0:
                        pred_instance_cat[key] = pred_instance[0][key]
                    else:
                        pred_instance_cat[key] = paddle.concat([
                            pred_instance[j][key]
                            for j in range(len(pred_instance))
                        ], 0)
                pred_instances_cat.append(pred_instance_cat)

            # 2D NMS and pick top-K.
            if self.do_nms:
                pred_instances = self.fcos2d_inference.nms_and_top_k(
                    pred_instances_cat, score_key)
            # print('pred_instances', pred_instances)
            pred_dicts = self.post_process(pred_instances, samples)
            # print('pred_dicts', pred_dicts)

            return {'preds': pred_dicts}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.shape[-2:]
            locations_per_level = self.compute_features_locations(
                h,
                w,
                self.input_strides[level],
                feature.dtype,
                offset=self.feature_locations_offset)
            locations.append(locations_per_level)
        return locations

    def compute_features_locations(self,
                                   h,
                                   w,
                                   stride,
                                   dtype='float32',
                                   offset="none"):
        shifts_x = paddle.arange(0, w * stride, step=stride, dtype=dtype)
        shifts_y = paddle.arange(0, h * stride, step=stride, dtype=dtype)
        shift_y, shift_x = paddle.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape([-1])
        shift_y = shift_y.reshape([-1])
        locations = paddle.stack((shift_x, shift_y), axis=1)
        if offset == "half":
            locations += stride // 2
        else:
            assert offset == "none"
        return locations

    def resize_instances(self, pred_boxes, height, width, image_size):
        image_size = [float(image_size[0]), float(image_size[1])]
        scale_x, scale_y = (width.cast('float32') / image_size[1],
                            height.cast('float32') / image_size[0])
        pred_boxes[:, 0::2] *= scale_x
        pred_boxes[:, 1::2] *= scale_y
        pred_boxes[:, 0::2] = pred_boxes[:, 0::2].clip(
            min=0.0, max=image_size[1])
        pred_boxes[:, 1::2] = pred_boxes[:, 1::2].clip(
            min=0.0, max=image_size[0])
        return pred_boxes

    def post_process(self, pred_instances, samples):
        pred_dicts = []
        for i, results_per_image in enumerate(pred_instances):
            if results_per_image['pred_boxes'].shape[0] == 0:
                data = Sample(samples["path"][i], samples["modality"][i])
                data.meta = SampleMeta(id=samples["meta"]['id'][i])
                pred_dicts.append(data)
                continue
            height = samples['image_sizes'][i, 0]
            width = samples['image_sizes'][i, 1]
            bboxes_2d = self.resize_instances(results_per_image['pred_boxes'],
                                              height, width,
                                              samples["data"].shape[-2:])
            bboxes_3d = []
            alpha = []
            for j in range(results_per_image['pred_boxes3d'].shape[0]):
                bbox_3d, alpha_ = self.convert_3d_box_to_kitti(
                    results_per_image['pred_boxes3d'][j:j + 1, :],
                    samples['meta']['camera_intrinsic'][i:i + 1, ...].inverse())
                bboxes_3d.append(bbox_3d)
                alpha.append(alpha_)
            data = Sample(samples["path"][i], samples["modality"][i])
            data.meta = SampleMeta(id=samples["meta"]['id'][i])
            data.calibs = samples["calibs"]
            bboxes_3d = np.array(bboxes_3d)
            labels = results_per_image['pred_classes'].numpy()
            confidences = results_per_image['scores_3d'].numpy()
            bboxes_2d = bboxes_2d.numpy()
            data.bboxes_3d = BBoxes3D(bboxes_3d)
            data.bboxes_3d.origin = [.5, 1., .5]
            data.bboxes_3d.coordmode = CoordMode.KittiCamera
            data.labels = labels
            data.confidences = confidences
            data.alpha = np.stack(alpha, 0)
            data.bboxes_2d = bboxes_2d
            pred_dicts.append(data)
        return pred_dicts

    def convert_3d_box_to_kitti(self, boxes3d, inv_intrinsics):
        quat = Quaternion(*boxes3d[:, :4].tolist()[0])
        ray = unproject_points2d(boxes3d[:, 4:6], inv_intrinsics)
        tvec = (ray * boxes3d[:, 6:7]).numpy()[0]
        sizes = boxes3d[:, 7:].numpy()[0]
        tvec += np.array([0., sizes[2] / 2.0, 0])
        inversion = Quaternion(axis=[1, 0, 0], radians=np.pi / 2).inverse
        quat = inversion * quat
        v_ = np.float64([[0, 0, 1], [0, 0, 0]])
        if quat.axis[2] > 0:
            v = self.pose(
                wxyz=Quaternion(axis=[0, 1, 0], radians=-quat.angle),
                tvec=tvec,
                v_=v_)
            rot_y = -quat.angle
        else:
            v = self.pose(
                wxyz=Quaternion(axis=[0, 1, 0], radians=quat.angle),
                tvec=tvec,
                v_=v_)
            rot_y = quat.angle
        v_ = v[:, ::2]
        theta = np.arctan2(abs(v_[1, 0]), abs(v_[1, 1]))
        alpha = rot_y + theta if v_[1, 0] < 0 else rot_y - theta
        # Bound from [-pi, pi]
        if alpha > np.pi:
            alpha -= 2.0 * np.pi
        elif alpha < -np.pi:
            alpha += 2.0 * np.pi
        alpha = np.around(alpha, decimals=2)  # KITTI precision

        return tvec.tolist() + [sizes[1], sizes[2], sizes[0]] + [rot_y], alpha

    def pose(self, wxyz, tvec, v_):
        quat = Quaternion(wxyz)
        matrix = quat.transformation_matrix
        matrix[:3, 3] = tvec
        X = np.hstack([v_, np.ones((len(v_), 1))]).T
        return (np.dot(matrix, X).T)[:, :3]

    def init_weight(self):
        if self.pretrained:
            checkpoint.load_pretrained_model(self, self.pretrained)
