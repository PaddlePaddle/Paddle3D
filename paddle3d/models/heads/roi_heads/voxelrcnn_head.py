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
"""
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/roi_heads/voxelrcnn_head.py
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""
import paddle
import paddle.nn as nn

from paddle3d.apis import manager
from paddle3d.models.common import generate_voxel2pinds, get_voxel_centers
from paddle3d.models.common.pointnet2_stack import \
    voxel_pool_modules as voxelpool_stack_modules
from paddle3d.models.heads.roi_heads.roi_head_base import RoIHeadBase
from paddle3d.models.layers import constant_init, xavier_normal_init


@manager.HEADS.add_component
class VoxelRCNNHead(RoIHeadBase):
    def __init__(self,
                 input_channels,
                 model_cfg,
                 point_cloud_range,
                 voxel_size,
                 num_class=1,
                 **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg["roi_grid_pool"]
        LAYER_cfg = self.pool_cfg["pool_layers"]
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        c_out = 0
        self.roi_grid_pool_layers = nn.LayerList()
        for src_name in self.pool_cfg["features_source"]:
            mlps = LAYER_cfg[src_name]["mlps"]
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name]["query_ranges"],
                nsamples=LAYER_cfg[src_name]["nsample"],
                radii=LAYER_cfg[src_name]["pool_radius"],
                mlps=mlps,
                pool_method=LAYER_cfg[src_name]["pool_method"],
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        GRID_SIZE = self.model_cfg["roi_grid_pool"]["grid_size"]
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg["shared_fc"].__len__()):
            shared_fc_list.extend([
                nn.Linear(
                    pre_channel,
                    self.model_cfg["shared_fc"][k],
                    bias_attr=False),
                nn.BatchNorm1D(self.model_cfg["shared_fc"][k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg["shared_fc"][k]

            if k != self.model_cfg["shared_fc"].__len__(
            ) - 1 and self.model_cfg["dp_ratio"] > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg["dp_ratio"]))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        for k in range(0, self.model_cfg['cls_fc'].__len__()):
            cls_fc_list.extend([
                nn.Linear(
                    pre_channel, self.model_cfg['cls_fc'][k], bias_attr=False),
                nn.BatchNorm1D(self.model_cfg['cls_fc'][k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg["cls_fc"][k]

            if k != self.model_cfg["cls_fc"].__len__(
            ) - 1 and self.model_cfg["dp_ratio"] > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg["dp_ratio"]))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(
            pre_channel, self.num_class, bias_attr=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg['reg_fc'].__len__()):
            reg_fc_list.extend([
                nn.Linear(
                    pre_channel, self.model_cfg['reg_fc'][k], bias_attr=False),
                nn.BatchNorm1D(self.model_cfg['reg_fc'][k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg['reg_fc'][k]

            if k != self.model_cfg['reg_fc'].__len__(
            ) - 1 and self.model_cfg["dp_ratio"] > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg["dp_ratio"]))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(
            pre_channel,
            self.box_coder.code_size * self.num_class,
            bias_attr=True)

        self.init_weights()

    def init_weights(self):
        for module_list in [
                self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers
        ]:
            for m in module_list.sublayers():
                if isinstance(m, nn.Linear):
                    xavier_normal_init(m.weight, reverse=True)
                    if m.bias is not None:
                        constant_init(m.bias, value=0)
                elif isinstance(m, nn.BatchNorm1D):
                    constant_init(m.weight, value=1)
                    constant_init(m.bias, value=0)

        self.cls_pred_layer.weight.set_value(
            paddle.normal(
                mean=0, std=0.01, shape=self.cls_pred_layer.weight.shape))
        constant_init(self.cls_pred_layer.bias, value=0)
        self.reg_pred_layer.weight.set_value(
            paddle.normal(
                mean=0, std=0.001, shape=self.reg_pred_layer.weight.shape))
        constant_init(self.reg_pred_layer.bias, value=0)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform',
                                           False)

        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg["grid_size"])  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.reshape([batch_size, -1, 3])

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = paddle.floor(
            (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) /
            self.voxel_size[0])
        roi_grid_coords_y = paddle.floor(
            (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) /
            self.voxel_size[1])
        roi_grid_coords_z = paddle.floor(
            (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) /
            self.voxel_size[2])
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = paddle.concat(
            [roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], axis=-1)

        batch_idx = paddle.zeros((batch_size, roi_grid_coords.shape[1], 1))
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = paddle.concat([batch_idx, roi_grid_coords], axis=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = paddle.full([
            batch_size,
        ],
                                         roi_grid_coords.shape[1],
                                         dtype='int32')

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg["features_source"]):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][
                    src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices().transpose([1, 0])
            cur_voxel_xyz = get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_strides=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range)
            cur_voxel_xyz_batch_cnt = paddle.zeros([
                batch_size,
            ],
                                                   dtype='int32')
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (
                    cur_coords[:, 0] == bs_idx).sum().astype(
                        cur_voxel_xyz_batch_cnt.dtype)
            # get voxel2point tensor
            v2p_ind_tensor = generate_voxel2pinds(cur_sp_tensors.shape,
                                                  cur_coords)

            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = paddle.floor(roi_grid_coords / cur_stride)
            cur_roi_grid_coords = paddle.concat(
                [batch_idx, cur_roi_grid_coords], axis=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.astype('int32')
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz,
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.reshape([-1, 3]),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.reshape([-1, 4]),
                features=cur_sp_tensors.values(),
                voxel2point_indices=v2p_ind_tensor)
            pooled_features = pooled_features.reshape(
                [-1, self.pool_cfg["grid_size"]**3,
                 pooled_features.shape[-1]])  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)

        ms_pooled_features = paddle.concat(pooled_features_list, axis=-1)

        return ms_pooled_features

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict,
            nms_config=self.model_cfg['nms_config']
            ['train' if self.training else 'test'])
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        # Box Refinement
        pooled_features = pooled_features.reshape(
            [pooled_features.shape[0], -1])
        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'],
                rois=batch_dict['rois'],
                cls_preds=rcnn_cls,
                box_preds=rcnn_reg)
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
