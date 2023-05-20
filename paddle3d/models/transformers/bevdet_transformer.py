# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle3d.ops import bev_pool_v2
from paddle3d.ops import bev_pool_v2_backward
from paddle.autograd import PyLayer
import paddle
import paddle.nn as nn
from paddle.vision.models.resnet import BasicBlock
import paddle.nn.functional as F
from paddle3d.apis import manager
from paddle3d.models.layers import param_init, reset_parameters, constant_init
from paddle3d.models.layers.layer_libs import SimConv


class QuickCumsumCuda(PyLayer):
    r"""BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`_
    """

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.cast('int32')
        depth = depth.cast('float32')
        feat = feat.cast('float32')
        ranks_depth = ranks_depth.cast('int32')
        ranks_feat = ranks_feat.cast('int32')
        interval_lengths = interval_lengths.cast('int32')
        interval_starts = interval_starts.cast('int32')

        out = bev_pool_v2.bev_pool_v2(depth, feat, ranks_depth, ranks_feat,
                                      ranks_bev, interval_lengths,
                                      interval_starts, bev_feat_shape)

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensor()

        order = ranks_feat.argsort()
        ranks_feat, ranks_depth, ranks_bev = \
            ranks_feat[order], ranks_depth[order], ranks_bev[order]
        kept = paddle.ones((ranks_bev.shape[0], ), dtype='bool')
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts_bp = paddle.where(kept)[0].cast("int32")
        interval_lengths_bp = paddle.zeros_like(interval_starts_bp)
        interval_lengths_bp[:-1] = interval_starts_bp[
            1:] - interval_starts_bp[:-1]
        interval_lengths_bp[-1] = ranks_bev.shape[0] - interval_starts_bp[-1]

        depth_grad = paddle.zeros_like(depth)
        feat_grad = paddle.zeros_like(feat)

        depth_grad, feat_grad = bev_pool_v2_backward.bev_pool_v2_bkwd(
            out_grad,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths_bp,
            interval_starts_bp,
        )
        return depth_grad, feat_grad, None, None, None, None


def bev_pool_v2_pyop(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                     bev_feat_shape, interval_starts, interval_lengths):
    x = QuickCumsumCuda.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              bev_feat_shape, interval_starts, interval_lengths)
    x = x.transpose((0, 3, 1, 2))
    return x


@manager.TRANSFORMERS.add_component
class LSSViewTransformer(nn.Layer):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`
    """

    def __init__(
            self,
            grid_config,
            input_size,
            downsample=16,
            in_channels=512,
            out_channels=64,
            accelerate=False,
    ):
        super(LSSViewTransformer, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.create_frustum(grid_config['depth'], input_size, downsample)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2D(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.
        """
        self.grid_lower_bound = paddle.to_tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = paddle.to_tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = paddle.to_tensor(
            [(cfg[1] - cfg[0]) / cfg[2] for cfg in [x, y, z]])

    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = paddle.arange(
            *depth_cfg, dtype='float32').reshape((-1, 1, 1)).expand((-1, H_feat,
                                                                     W_feat))
        self.D = d.shape[0]
        x = paddle.linspace(
            0, W_in - 1, W_feat, dtype='float32').reshape(
                (1, 1, W_feat)).expand((self.D, H_feat, W_feat))
        y = paddle.linspace(
            0, H_in - 1, H_feat, dtype='float32').reshape(
                (1, H_feat, 1)).expand((self.D, H_feat, W_feat))

        # D x H x W x 3
        self.frustum = paddle.stack((x, y, d), axis=-1)

    def get_lidar_coor(self, rots, trans, cam2imgs, post_rots, post_trans, bda):
        """Calculate the locations of the frustum points in the lidar
        """
        B, N, _ = trans.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.cast(rots.dtype) - post_trans.reshape(
            (B, N, 1, 1, 1, 3))
        points_ori = points

        # hacky way to solve 7-dim matmul speed problem
        _, _, D, H, W, _ = points.shape
        post_rots = paddle.inverse(post_rots).reshape(
            (B, N, 1, 1, 1, 3, 3)).reshape((B * N, 1, 1, 1, 3, 3)).expand(
                (-1, D, H, W, -1, -1)).reshape((-1, 3, 3))
        y = points.reshape((B, N, D, H, W, 3, 1)).reshape((-1, 3, 1))

        points = post_rots.matmul(y)
        points = points.reshape((B, N, D, H, W, 3, 1))

        # cam_to_ego
        # paddle concat only support input dim < 7
        points = points.reshape(points.shape[:-1])
        points = paddle.concat(
            (points[..., :2] * points[..., 2:3], points[..., 2:3]), axis=5)
        points = points.reshape((*points.shape, 1))
        cam2imgs = cam2imgs.cast("float32")
        combine = rots.matmul(y=paddle.inverse(cam2imgs))
        # hacky way to solve 7-dim matmul speed problem
        combine = combine.reshape((B, N, 1, 1, 1, 3, 3)).reshape(
            (B * N, 1, 1, 1, 3, 3)).expand((-1, D, H, W, -1, -1)).reshape(
                (-1, 3, 3))
        y = points.reshape((-1, 3, 1))
        points = combine.matmul(y).reshape((B, N, D, H, W, 3))
        points += trans.reshape((B, N, 1, 1, 1, 3))

        # hacky way to solve 7-dim matmul speed problem
        bda = bda.reshape((B, 1, 1, 1, 1, 3, 3)).reshape(
            (B, 1, 1, 1, 1, 9)).expand((-1, N, D, H, W, -1)).reshape(
                (-1, 9)).reshape((-1, 3, 3))
        y = points.reshape((B, N, D, H, W, 3, 1)).reshape((-1, 3, 1))
        points = bda.matmul(y).reshape((B, N, D, H, W, 3))
        return points

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        self.ranks_bev = ranks_bev.cast('int32')
        self.ranks_feat = ranks_feat.cast('int32')
        self.ranks_depth = ranks_depth.cast('int32')
        self.interval_starts = interval_starts.cast('int32')
        self.interval_lengths = interval_lengths.cast('int32')

    def voxel_pooling_v2(self, coor, depth, feat):
        B = coor.shape[0]
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = paddle.zeros(shape=[
                B, feat.shape[1],
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).cast(feat.dtype)

            return dummy
        feat = feat.transpose([0, 2, 3, 1])
        bev_feat_shape = (B, int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])
        bev_feat = bev_pool_v2_pyop(depth, feat, ranks_depth, ranks_feat,
                                    ranks_bev, bev_feat_shape, interval_starts,
                                    interval_lengths)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = paddle.arange(0, num_points, dtype='int')
        ranks_feat = paddle.arange(0, num_points // D, dtype='int')
        ranks_feat = ranks_feat.reshape((B, N, 1, H, W))
        ranks_feat = ranks_feat.expand((B, N, D, H, W)).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.cast(coor.dtype)) /
                self.grid_interval.cast(coor.dtype))
        coor = coor.cast('int64').reshape((num_points, 3))
        batch_idx = paddle.arange(0, B).reshape((B, 1)).expand(
            (B, num_points // B)).reshape((num_points, 1)).cast(coor.dtype)
        coor = paddle.concat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = paddle.ones((ranks_bev.shape[0], ), dtype=paddle.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = paddle.where(kept)[0].cast('int32').squeeze(-1)
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = paddle.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.cast('int32'), ranks_depth.cast(
            'int32'), ranks_feat.cast('int32'), interval_starts.cast(
                'int32'), interval_lengths.cast('int32')

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_lidar_coor(*input[1:7])
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.transpose([0, 2, 3, 1])
            bev_feat_shape = (B, int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])
            bev_feat = bev_pool_v2_pyop(
                depth, feat, self.ranks_depth, self.ranks_feat, self.ranks_bev,
                bev_feat_shape, self.interval_starts, self.interval_lengths)

        else:
            coor = self.get_lidar_coor(*input[1:7])
            bev_feat = self.voxel_pooling_v2(coor, depth, tran_feat)
        return bev_feat, depth

    def view_transform(self, input, depth, tran_feat):
        if self.accelerate:
            self.pre_compute(input)
        ret = self.view_transform_core(input, depth, tran_feat)
        return ret

    def forward(self, input):
        """Transform image-view feature into bird-eye-view feature.
        """
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x)

        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None


class _ASPPModule(nn.Layer):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2D(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias_attr=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                param_init.kaiming_normal_init(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                param_init.constant_init(m.weight, value=1)
                param_init.constant_init(m.bias, value=0)


class ASPP(nn.Layer):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2D):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Conv2D(inplanes, mid_channels, 1, stride=1, bias_attr=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2D(
            int(mid_channels * 5), mid_channels, 1, bias_attr=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x = paddle.concat((x1, x2, x3, x4, x5), axis=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                param_init.kaiming_normal_init(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                param_init.constant_init(m.weight, value=1)
                param_init.constant_init(m.bias, value=0)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(27, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                reset_parameters(m)


class SELayer(nn.Layer):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2D(channels, channels, 1, bias_attr=None)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2D(channels, channels, 1, bias_attr=None)
        self.gate = gate_layer()
        self.init_weights()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

    def init_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                reset_parameters(m)


class SimSPPF(nn.Layer):
    '''Simplified SPPF with ReLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2D(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(paddle.concat([x, y1, y2, self.m(y2)], 1))


class MSDepthNet(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=False,
                 use_aspp=False,
                 use_sppf=True):
        super(MSDepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2D(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
        )
        self.context_conv = nn.Conv2D(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1D(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)
        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
        ]

        depth_conv_list.append(SimSPPF(mid_channels, mid_channels))

        if use_aspp:
            raise NotImplementedError

        if use_dcn:  # todo
            raise NotImplementedError

        self.depth_conv_low = nn.Sequential(*depth_conv_list)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        depth_conv_list_mid = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]

        depth_conv_list_mid.append(
            nn.Conv2D(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv_mid = nn.Sequential(*depth_conv_list_mid)

        self._init_weight()

    def _init_weight(self):
        self.depth_conv_low.apply(param_init.init_weight)
        self.depth_conv_mid.apply(param_init.init_weight)

        self.reduce_conv.apply(param_init.init_weight)
        self.depth_se.apply(param_init.init_weight)
        self.depth_mlp.apply(param_init.init_weight)
        self.context_mlp.apply(param_init.init_weight)
        self.context_se.apply(param_init.init_weight)
        self.context_conv.apply(param_init.init_weight)
        param_init.init_weight(self.bn)

    def forward(self, x_high, x_mid, x_low, mlp_input):
        mlp_input.stop_gradient = False
        mlp_input = self.bn(mlp_input.reshape((-1, mlp_input.shape[-1])))
        x_high = self.reduce_conv(x_high)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x_low, depth_se)
        depth = self.depth_conv_low(depth)
        depth = self.up(depth)
        depth = x_mid + depth
        depth = self.depth_conv_mid(depth)

        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x_high, context_se)
        context = self.context_conv(context)

        depth = self.up(depth)

        return depth, context


class DepthNet(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=False,
                 use_aspp=True):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2D(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
        )
        self.context_conv = nn.Conv2D(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm1D(27, data_format='NC')
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)
        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            depth_conv_list.append(ASPP(mid_channels, mid_channels))

        if use_dcn:  # TODO support dcn
            print("dcn not supported yet")
            raise NotImplementedError

        depth_conv_list.append(
            nn.Conv2D(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self._init_weight()

    def forward(self, x, mlp_input):
        mlp_input.stop_gradient = False  # TODO stop gridient problem
        mlp_input = self.bn(mlp_input.reshape((-1, mlp_input.shape[-1])))
        self.mlp_input = mlp_input
        x = self.reduce_conv(x)

        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return paddle.concat([depth, context], axis=1)

    def _init_weight(self):
        print('init_depthnet_weight')
        self.depth_conv[0].apply(param_init.init_weight)
        self.depth_conv[1].apply(param_init.init_weight)
        self.depth_conv[2].apply(param_init.init_weight)
        self.depth_conv[-1].apply(param_init.init_weight)

        self.reduce_conv.apply(param_init.init_weight)
        self.depth_se.apply(param_init.init_weight)
        self.depth_mlp.apply(param_init.init_weight)
        self.context_mlp.apply(param_init.init_weight)
        self.context_se.apply(param_init.init_weight)
        self.context_conv.apply(param_init.init_weight)
        param_init.init_weight(self.bn)
        print('finish init depth weight!')


@manager.TRANSFORMERS.add_component
class LSSViewTransformerBEVDepth(LSSViewTransformer):
    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformerBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.out_channels, self.D, **depthnet_cfg)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        B, N, _, _ = rot.shape
        bda = bda.reshape((B, 1, 3, 3)).tile([1, N, 1, 1])
        intrin = intrin.cast("float32")
        mlp_input = paddle.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ],
                                 axis=-1)
        sensor2ego = paddle.concat([rot, tran.reshape((B, N, 3, 1))],
                                   axis=-1).reshape((B, N, -1))
        mlp_input = paddle.concat([mlp_input, sensor2ego], axis=-1)
        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.reshape(
            (B * N, H // self.downsample, self.downsample, W // self.downsample,
             self.downsample, 1))
        gt_depths = gt_depths.transpose((0, 1, 3, 5, 2, 4))
        gt_depths = gt_depths.reshape((-1, self.downsample * self.downsample))
        gt_depths_tmp = paddle.where(
            gt_depths == 0.0, 1e5 * paddle.ones(
                (gt_depths.shape), dtype=gt_depths.dtype), gt_depths)
        gt_depths = paddle.min(gt_depths_tmp, axis=-1)
        gt_depths = gt_depths.reshape((B * N, H // self.downsample,
                                       W // self.downsample))

        gt_depths = (gt_depths - (
            self.grid_config['depth'][0] - self.grid_config['depth'][2])
                     ) / self.grid_config['depth'][2]

        gt_depths = paddle.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths,
            paddle.zeros(gt_depths.shape, dtype=gt_depths.dtype))
        gt_depths = F.one_hot(
            gt_depths.cast("int64"), num_classes=self.D + 1).reshape(
                (-1, self.D + 1))[:, 1:]
        return gt_depths.cast("float32")

    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.transpose((0, 2, 3, 1))
        depth_preds = depth_preds.reshape((-1, self.D))
        fg_mask = paddle.max(depth_labels, axis=1) > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        depth_loss = F.binary_cross_entropy(
            depth_preds,
            depth_labels,
            reduction='none',
        ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.reshape((B * N, C, H, W))
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = F.softmax(depth_digit, axis=1)
        ret = self.view_transform(input, depth, tran_feat)
        return ret


@manager.TRANSFORMERS.add_component
class MSLSSViewTransformerBEVDepth(LSSViewTransformer):
    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(MSLSSViewTransformerBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = MSDepthNet(self.in_channels, self.in_channels,
                                    self.out_channels, self.D, **depthnet_cfg)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        B, N, _, _ = rot.shape
        bda = bda.reshape((B, 1, 3, 3)).tile([1, N, 1, 1])
        intrin = intrin.cast("float32")
        mlp_input = paddle.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ],
                                 axis=-1)
        sensor2ego = paddle.concat([rot, tran.reshape((B, N, 3, 1))],
                                   axis=-1).reshape((B, N, -1))
        mlp_input = paddle.concat([mlp_input, sensor2ego], axis=-1)
        return mlp_input

    def get_downsampled_gt_depth(self, gt_depths):
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.reshape(
            (B * N, H // self.downsample, self.downsample, W // self.downsample,
             self.downsample, 1))
        gt_depths = gt_depths.transpose((0, 1, 3, 5, 2, 4))
        gt_depths = gt_depths.reshape((-1, self.downsample * self.downsample))
        gt_depths_tmp = paddle.where(
            gt_depths == 0.0, 1e5 * paddle.ones(
                (gt_depths.shape), dtype=gt_depths.dtype), gt_depths)
        gt_depths = paddle.min(gt_depths_tmp, axis=-1)
        gt_depths = gt_depths.reshape((B * N, H // self.downsample,
                                       W // self.downsample))

        gt_depths = (gt_depths - (
            self.grid_config['depth'][0] - self.grid_config['depth'][2])
                     ) / self.grid_config['depth'][2]

        gt_depths = paddle.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths,
            paddle.zeros(gt_depths.shape, dtype=gt_depths.dtype))
        gt_depths = F.one_hot(
            gt_depths.cast("int64"), num_classes=self.D + 1).reshape(
                (-1, self.D + 1))[:, 1:]
        return gt_depths.cast("float32")

    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.transpose((0, 2, 3, 1))
        depth_preds = depth_preds.reshape((-1, self.D))
        fg_mask = paddle.max(depth_labels, axis=1) > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        depth_loss = F.binary_cross_entropy(
            depth_preds,
            depth_labels,
            reduction='none',
        ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N = rots.shape[:2]
        _, C, H, W = x[0].shape
        x_feat = x[0].reshape([B, N, C, H, W])
        input = [
            x_feat, rots, trans, intrins, post_rots, post_trans, bda, mlp_input
        ]
        depth_digit, tran_feat = self.depth_net(x[0], x[1], x[2], mlp_input)
        depth = F.softmax(depth_digit, axis=1)

        ret = self.view_transform(input, depth, tran_feat)
        return ret
