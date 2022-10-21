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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec

from paddle3d.apis import manager
from paddle3d.models.common import class_agnostic_nms
from paddle3d.models.layers import ConvBNReLU
from paddle3d.utils import checkpoint
from paddle3d.utils.logger import logger

from .bev import BEV
from .f2v import FrustumToVoxel
from .ffe import FFE


@manager.MODELS.add_component
class CADDN(nn.Layer):
    """
    """

    def __init__(self,
                 backbone_3d,
                 bev_cfg,
                 dense_head,
                 class_head,
                 ffe_cfg,
                 f2v_cfg,
                 disc_cfg,
                 map_to_bev_cfg,
                 post_process_cfg,
                 pretrained=None):
        super().__init__()
        self.backbone_3d = backbone_3d
        self.class_head = class_head
        self.ffe = FFE(ffe_cfg, disc_cfg=disc_cfg)
        self.map_to_bev = ConvBNReLU(**map_to_bev_cfg)
        self.backbone_2d = BEV(**bev_cfg)
        self.dense_head = dense_head
        self.f2v = FrustumToVoxel(**f2v_cfg, disc_cfg=disc_cfg)
        self.post_process_cfg = post_process_cfg
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, data):
        images = data["images"]
        if not self.training:
            b, c, h, w = paddle.shape(images)
            data["batch_size"] = b
            data["image_shape"] = paddle.concat([h, w]).unsqueeze(0)
        # ffe
        image_features = self.backbone_3d(images)

        depth_logits = self.class_head(image_features, data["image_shape"])
        data = self.ffe(image_features[0], depth_logits, data)

        #   frustum_to_voxel
        data = self.f2v(data)

        # map_to_bev
        # voxel_features = voxel_features.reshape([])
        voxel_features = data["voxel_features"]
        bev_features = voxel_features.flatten(
            start_axis=1, stop_axis=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        b, c, h, w = paddle.shape(bev_features)
        bev_features = bev_features.reshape(
            [b, self.map_to_bev._conv._in_channels, h, w])
        bev_features = self.map_to_bev(
            bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        data["spatial_features"] = bev_features

        # backbone_2d
        data = self.backbone_2d(data)
        predictions = self.dense_head(data)

        if not self.training:
            return self.post_process(predictions)
        else:
            loss = self.get_loss(predictions)
            return loss

    def get_loss(self, predictions):
        disp_dict = {}

        loss_rpn, tb_dict_rpn = self.dense_head.get_loss()
        loss_depth, tb_dict_depth = self.ffe.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_depth': loss_depth.item(),
            **tb_dict_rpn,
            **tb_dict_depth
        }

        loss = loss_rpn + loss_depth

        losses = {"loss": loss, "tb_dict": tb_dict, "disp_dict": disp_dict}
        return losses

    def post_process(self, batch_dict):
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
        batch_size = 1  # batch_dict['batch_size']
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

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = F.sigmoid(cls_preds)
            else:
                cls_preds = [
                    x[batch_mask] for x in batch_dict['batch_cls_preds']
                ]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [F.sigmoid(x) for x in cls_preds]

            label_preds = paddle.argmax(cls_preds, axis=-1) + 1.0
            cls_preds = paddle.max(cls_preds, axis=-1)
            selected_score, selected_label, selected_box = class_agnostic_nms(
                box_scores=cls_preds,
                box_preds=box_preds,
                label_preds=label_preds,
                nms_config=self.post_process_cfg['nms_config'],
                score_thresh=self.post_process_cfg['score_thresh'])
            record_dict = paddle.concat([
                selected_score.unsqueeze(1), selected_box,
                selected_label.unsqueeze(1)
            ],
                                        axis=1)

            record_dict = {
                'pred_boxes': selected_box,
                'pred_scores': selected_score,
                'pred_labels': selected_label
            }

            pred_dicts.append(record_dict)
        return {'preds': pred_dicts}

    def init_weight(self):
        if self.pretrained:
            checkpoint.load_pretrained_model(self, self.pretrained)

    def export(self, save_dir: str, **kwargs):
        self.export_model = True
        save_path = os.path.join(save_dir, 'caddn')
        input_spec = [{
            "images":
            InputSpec(shape=[None, 3, None, None], name="images"),
            "trans_lidar_to_cam":
            InputSpec(shape=[None, 4, 4], name='trans_lidar_to_cam'),
            "trans_cam_to_img":
            InputSpec(shape=[None, 3, 4], name='trans_cam_to_img'),
        }]

        paddle.jit.to_static(self, input_spec=input_spec)
        paddle.jit.save(self, save_path, input_spec=[input_spec])

        logger.info("Exported model is saved in {}".format(save_dir))
