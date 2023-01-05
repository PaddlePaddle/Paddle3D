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

# ------------------------------------------------------------------------
# Modified from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy

import numpy as np
import paddle
import paddle.nn as nn

from paddle3d.apis import manager


@manager.TRANSFORMER_ENCODERS.add_component
class BEVFormerEncoder(nn.Layer):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 transformerlayers,
                 num_layers,
                 point_cloud_range=None,
                 num_points_in_pillar=4,
                 return_intermediate=False,
                 dataset_type='nuscenes',
                 **kwargs):
        super(BEVFormerEncoder, self).__init__()
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.point_cloud_range = point_cloud_range
        self.layers = nn.LayerList()
        for i in range(num_layers):
            layer_name = transformerlayers[i].pop('type_name')
            encoder_layer = manager.TRANSFORMER_ENCODER_LAYERS.components_dict[
                layer_name]
            params = transformerlayers[i]
            self.layers.append(encoder_layer(**params))

    @staticmethod
    def get_reference_points(H,
                             W,
                             Z=8,
                             num_points_in_pillar=4,
                             dim='3d',
                             bs=1,
                             dtype=paddle.float32):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = paddle.linspace(
                0.5, Z - 0.5, num_points_in_pillar,
                dtype=paddle.float32).cast(dtype).reshape([-1, 1, 1]).expand(
                    [num_points_in_pillar, H, W]) / Z
            xs = paddle.linspace(
                0.5, W - 0.5, W, dtype=paddle.float32).reshape([
                    1, 1, W
                ]).cast(dtype).expand([num_points_in_pillar, H, W]) / W
            ys = paddle.linspace(
                0.5, H - 0.5, H, dtype=paddle.float32).reshape([
                    1, H, 1
                ]).cast(dtype).expand([num_points_in_pillar, H, W]) / H
            ref_3d = paddle.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.transpose([0, 3, 1,
                                       2]).flatten(2).transpose([0, 2, 1])
            ref_3d = ref_3d[None].tile([bs, 1, 1, 1])
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = paddle.meshgrid(
                paddle.linspace(0.5, H - 0.5, H, dtype=paddle.float32),
                paddle.linspace(0.5, W - 0.5, W, dtype=paddle.float32))
            ref_y = ref_y.cast(dtype).reshape([-1])[None] / H
            ref_x = ref_x.cast(dtype).reshape([-1])[None] / W
            ref_2d = paddle.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.tile([bs, 1, 1]).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    def point_sampling(self, reference_points, point_cloud_range, img_metas):
        reference_points = reference_points.cast(paddle.float32)
        if not getattr(self, 'export_model', False):
            lidar2img = []
            for img_meta in img_metas:
                lidar2img.append(paddle.stack(img_meta['lidar2img']))
            lidar2img = paddle.stack(lidar2img)  # (B, N, 4, 4)
        else:
            lidar2img = img_metas[0]['lidar2img']
        lidar2img = lidar2img.cast(paddle.float32)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]

        reference_points = paddle.concat(
            (reference_points, paddle.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.transpose([1, 0, 2, 3])
        D, B, num_query = reference_points.shape[:3]
        num_cam = lidar2img.shape[1]

        reference_points = reference_points.reshape(
            [D, B, 1, num_query, 4]).tile([1, 1, num_cam, 1, 1]).unsqueeze(-1)

        lidar2img = lidar2img.reshape([1, B, num_cam, 1, 4,
                                       4]).tile([D, 1, 1, num_query, 1, 1])

        # np.save("e_lidar2img.npy", lidar2img.numpy())
        # np.save("e_reference_points.npy", reference_points.numpy())
        reference_points_cam = paddle.matmul(
            lidar2img.cast(paddle.float32),
            reference_points.cast(paddle.float32)).squeeze(-1)
        # np.save("e_points_cam.npy", reference_points_cam.numpy())
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / paddle.maximum(
            reference_points_cam[..., 2:3],
            paddle.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        reference_points_cam = reference_points_cam.transpose([2, 1, 3, 0, 4])
        bev_mask = bev_mask.transpose([2, 1, 3, 0, 4]).squeeze(-1)

        return reference_points_cam, bev_mask

    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.point_cloud_range[5] - self.point_cloud_range[2],
            self.num_points_in_pillar,
            dim='3d',
            bs=bev_query.shape[1],
            dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim='2d',
            bs=bev_query.shape[1],
            dtype=bev_query.dtype)
        # np.save("e_ref_2d.npy", ref_2d.numpy())

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.point_cloud_range, kwargs['img_metas'])

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        # TODO(qianhui): fix this clone bugs: paddle equal means clone but torch not
        #shift_ref_2d = ref_2d
        #shift_ref_2d += shift[:, None, None, :]
        ref_2d += shift[:, None, None, :]
        # np.save("e_shift_ref_2d.npy", ref_2d.numpy())
        # np.save("e_ref_2dref_2d.npy", ref_2d.numpy())

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.transpose([1, 0, 2])
        bev_pos = bev_pos.transpose([1, 0, 2])
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        '''
        if prev_bev is not None:
            prev_bev = prev_bev.transpose([1, 0, 2])
            prev_bev = paddle.stack(
                [prev_bev, bev_query], 1).reshape([bs*2, len_bev, -1])
            # TODO(qianhui): fix this clone bugs: paddle equal means clone but torch not
            #hybird_ref_2d = paddle.stack([shift_ref_2d, ref_2d], 1).reshape(
            hybird_ref_2d = paddle.stack([ref_2d, ref_2d], 1).reshape(
                [bs*2, len_bev, num_bev_level, 2])
        else:
            hybird_ref_2d = paddle.stack([ref_2d, ref_2d], 1).reshape(
                [bs*2, len_bev, num_bev_level, 2])
        '''
        prev_bev = prev_bev.transpose([1, 0, 2])
        valid_prev_bev = prev_bev.cast('bool').any().cast('int32')
        prev_bev = prev_bev * valid_prev_bev + bev_query * (1 - valid_prev_bev)
        prev_bev = paddle.stack([prev_bev, bev_query],
                                1).reshape([bs * 2, len_bev, -1])
        hybird_ref_2d = paddle.stack([ref_2d, ref_2d], 1).reshape(
            [bs * 2, len_bev, num_bev_level, 2])

        # np.save("e_bev_query.npy", bev_query.numpy())
        # np.save("e_key.npy", key.numpy())
        # np.save("e_value.npy", value.numpy())
        # np.save("e_bev_posbev_pos.npy", bev_pos.numpy())
        # np.save("e_hybird_ref_2d.npy", hybird_ref_2d.numpy())
        # np.save("e_ref_3d.npy", ref_3d.numpy())
        # np.save("e_spatial_shapes.npy", spatial_shapes.numpy())
        # np.save("e_reference_points_cam.npy", reference_points_cam.numpy())
        # np.save("e_bev_mask.npy", bev_mask.numpy())
        # np.save("e_prev_bev.npy", prev_bev.numpy())
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)
            # np.save("e_output_{}.npy".format(lid), output.numpy())

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return paddle.stack(intermediate)

        return output
