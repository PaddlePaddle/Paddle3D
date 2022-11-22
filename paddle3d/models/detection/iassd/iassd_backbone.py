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

# This code is based on https://github.com/yifanzhang713/IA-SSD/blob/main/pcdet/models/backbones_3d/IASSD_backbone.py

import numpy as np
import paddle
import paddle.nn as nn

from paddle3d.apis import manager

from .iassd_modules import SAModuleMSG_WithSampling, Vote_layer

__all__ = ["IASSD_Backbone"]


@manager.BACKBONES.add_component
class IASSD_Backbone(nn.Layer):
    """Backbone of IA-SSD

    Args:
        npoint_list (List[int]): num of sampled points in each layer.
        sample_method_list (List[str]): sample method in each layer.
        radius_list (List[List[float]]): radius params in multi-scale SA layer.
        nsample_list (List[List[int]]): num of sampled points in multi-scale SA layer.
        mlps (List[List[int]]): hidden dim of mlps in SA layer.
        layer_types (List[str]): type of layer, SA or Vote layer in IA-SSD.
        dilated_group (List[bool]): not implemented, set to False in default.
        aggregation_mlps (List[List[int]]): hidden dim of aggregation mlps, used to aggregate the outputs of multi-scale SA layer.
        confidence_mlps (List[List[int]]): hidden dim of confidence mlps, used to predict classes of each point.
        layer_input (List[int]): index of layer input, determine which layer's outputs feeded in current layer.
        ctr_index (List[int]): index of centroid, determine which layer's outpus used in centroid prediction.
        max_translate_range (List[float]): limit the max range of predicted offset in Vote layer.
        input_channle (int): input pointcloud feature dim.
        num_classes (int): number of classes.
    """

    def __init__(
            self,
            npoint_list,
            sample_method_list,
            radius_list,
            nsample_list,
            mlps,
            layer_types,
            dilated_group,
            aggregation_mlps,
            confidence_mlps,
            layer_input,
            ctr_index,
            max_translate_range,
            input_channel,
            num_classes,
    ):
        super().__init__()
        self.npoint_list = npoint_list
        self.sample_method_list = sample_method_list
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlps = mlps
        self.layer_types = layer_types
        self.dilated_group = dilated_group
        self.aggregation_mlps = aggregation_mlps
        self.confidence_mlps = confidence_mlps
        self.layer_input = layer_input
        self.ctr_idx_list = ctr_index
        self.max_translate_range = max_translate_range
        self.export_model = False

        channel_in = input_channel - 3
        channel_out_list = [channel_in]
        self.SA_modules = nn.LayerList()

        for k in range(len(self.nsample_list)):
            channel_in = channel_out_list[self.layer_input[k]]

            if self.layer_types[k] == "SA_Layer":
                mlps = self.mlps[k].copy()
                channel_out = 0
                for idx in range(len(mlps)):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]

                if self.aggregation_mlps and self.aggregation_mlps[k]:
                    aggregation_mlp = self.aggregation_mlps[k].copy()
                    if len(aggregation_mlp) == 0:
                        aggregation_mlp = None
                    else:
                        channel_out = aggregation_mlp[-1]
                else:
                    aggregation_mlp = None

                if self.confidence_mlps and self.confidence_mlps[k]:
                    confidence_mlp = self.confidence_mlps[k].copy()
                    if len(confidence_mlp) == 0:
                        confidence_mlp = None
                else:
                    confidence_mlp = None

                self.SA_modules.append(
                    SAModuleMSG_WithSampling(
                        npoint=self.npoint_list[k],
                        sample_range=-1,
                        sample_type=self.sample_method_list[k],
                        radii=self.radius_list[k],
                        nsamples=self.nsample_list[k],
                        mlps=mlps,
                        use_xyz=True,
                        dilated_group=self.dilated_group[k],
                        aggregation_mlp=aggregation_mlp,
                        confidence_mlp=confidence_mlp,
                        num_classes=num_classes,
                    ))

            elif self.layer_types[k] == "Vote_Layer":
                self.SA_modules.append(
                    Vote_layer(
                        mlp_list=self.mlps[k],
                        pre_channel=channel_out_list[self.layer_input[k]],
                        max_translate_range=self.max_translate_range,
                    ))

            channel_out_list.append(channel_out)

        self.num_point_features = channel_out

    def forward(self, batch_dict):
        """
        Args:
            batch_dict: input dict of batched point data and box annos.
                data: (num_points * B, 3 + C), input point cloud, C is feature dim.
                batch_size: B.
                num_points: number of points in single point cloud
        Return:
            batch_dict: add new fileds int to input batch_dict
        """

        points = batch_dict["data"]
        # for export only
        if self.export_model:
            batch_dict["batch_size"] = 1
            points = self.stack_batch_idx_to_points(
                points, num_points=16384)  # 16384 for kitti
        batch_size = batch_dict["batch_size"]

        batch_idx, xyz, features = self.break_up_pc(points)
        xyz = xyz.reshape([batch_size, -1, 3])
        features = (features.reshape([
            batch_size, -1, features.shape[-1]
        ]).transpose([0, 2, 1]) if features is not None else None)

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        encoder_coords = [
            paddle.concat([batch_idx.reshape([batch_size, -1, 1]), xyz],
                          axis=-1)
        ]

        # yapf: disable
        li_cls_pred = None
        var_dict = dict()
        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_input[i]]
            feature_input = encoder_features[self.layer_input[i]]

            if self.layer_types[i] == "SA_Layer":
                ctr_xyz = (encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None)
                li_xyz, li_features, li_cls_pred = self.SA_modules[i](
                    xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz)

            elif self.layer_types[i] == "Vote_Layer":
                li_xyz, li_features, xyz_select, ctr_offsets = self.SA_modules[
                    i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_select
                var_dict["ctr_offsets"] = ctr_offsets
                var_dict["centers"] = centers
                var_dict["centers_origin"] = centers_origin

                center_origin_batch_idx = batch_idx.reshape([batch_size, -1])[:, :centers_origin.shape[1]]
                encoder_coords.append(
                    paddle.concat(
                        [
                            center_origin_batch_idx[..., None].astype("float32"),
                            centers_origin.reshape([batch_size, -1, 3])
                        ],
                        axis=-1
                    ))
            encoder_xyz.append(li_xyz)
            li_batch_idx = batch_idx.reshape([batch_size, -1])[:, :li_xyz.shape[1]]
            encoder_coords.append(
                paddle.concat(
                    [
                        li_batch_idx[..., None].astype("float32"),
                        li_xyz.reshape([batch_size, -1, 3])
                    ],
                    axis=-1
                ))
            encoder_features.append(li_features)
            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.reshape([batch_size, -1])[:, :li_cls_pred.shape[1]]
                sa_ins_preds.append(
                    paddle.concat(
                        [
                            li_cls_batch_idx[..., None].astype("float32"),
                            li_cls_pred.reshape([batch_size, -1, li_cls_pred.shape[-1]])
                        ],
                        axis=-1
                    ))
            else:
                sa_ins_preds.append([])

        ctr_batch_idx = batch_idx.reshape([batch_size, -1])[:, :encoder_xyz[-1].shape[1]]
        ctr_batch_idx = ctr_batch_idx.reshape([-1])

        batch_dict["ctr_offsets"] = paddle.concat(
            [
                ctr_batch_idx[:, None].astype("float32"),
                var_dict["ctr_offsets"].reshape([-1, 3])
            ],
            axis=1
        )

        batch_dict["centers"] = paddle.concat(
            [
                ctr_batch_idx[:, None].astype("float32"),
                var_dict["centers"].reshape([-1, 3])
            ],
            axis=1
        )

        batch_dict["centers_origin"] = paddle.concat(
            [
                ctr_batch_idx[:, None].astype("float32"),
                var_dict["centers_origin"].reshape([-1, 3])
            ],
            axis=1
        )
        # yapf: enable

        center_features = (encoder_features[-1].transpose([0, 2, 1]).reshape(
            [-1, encoder_features[-1].shape[1]]))
        batch_dict["centers_features"] = center_features
        batch_dict["ctr_batch_idx"] = ctr_batch_idx
        batch_dict["encoder_xyz"] = encoder_xyz
        batch_dict["encoder_coords"] = encoder_coords
        batch_dict["sa_ins_preds"] = sa_ins_preds
        batch_dict["encoder_features"] = encoder_features

        return batch_dict

    def break_up_pc(self, pc):
        """break up point cloud into xyz + point_feature
        Args:
            pc: (num_points * B, C)
        Return:
            batch_idx: (num_points * B, 1), batch index of input data
            xyz: (num_points * B, 3), coordinates of points
            features: (num_points * B, C), features of points
        """
        batch_idx = pc[:, 0]  # (B*N, 1)
        xyz = pc[:, 1:4]  # (B*N, 3)
        features = pc[:, 4:] if pc.shape[
            -1] > 4 else None  # (B*N, C) C=1 for intensity
        return batch_idx, xyz, features

    def stack_batch_idx_to_points(self, points, num_points=16384):
        batch_idx = paddle.zeros([num_points, 1], dtype='float32')
        points = paddle.concat([batch_idx, points], axis=-1)
        return points
