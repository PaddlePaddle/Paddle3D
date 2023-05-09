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

# """
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
# Authors: Jonah Philion and Sanja Fidler
# """

# This code is based on https://github.com/ADLab-AutoDrive/BEVFusion/blob/main/mmdet3d/models/detectors/cam_stream_lss.py

import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.vision.models import resnet18

from paddle3d.models.layers.param_init import constant_init, reset_parameters

__all__ = ['LiftSplatShoot']


class Up(nn.Layer):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False), nn.BatchNorm2D(out_channels), nn.ReLU(),
            nn.Conv2D(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False), nn.BatchNorm2D(out_channels), nn.ReLU())

    def forward(self, x1, x2):
        x1 = F.interpolate(
            x1, x2.shape[2:], mode='bilinear', align_corners=True)
        x1 = paddle.concat([x2, x1], axis=1)
        return self.conv(x1)


class BevEncode(nn.Layer):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False)
        self.conv1 = nn.Conv2D(
            inC, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2D(256, 128, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


def gen_dx_bx(xbound, ybound, zbound):
    dx = paddle.to_tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = paddle.to_tensor(
        [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = [int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]]

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    # TODO(liuxiao): cumsum will return wrong anser when input tensor has large shape,
    # transpose to the last aixs to sum will temporialy fix this problem.
    x = x.transpose([1, 0]).cumsum(1).transpose([1, 0])
    kept = paddle.ones([x.shape[0]], dtype='bool')
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = paddle.concat([x[:1], x[1:] - x[:-1]])

    return x, geom_feats


class QuickCumsum(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.transpose([1, 0]).cumsum(1).transpose([1, 0])
        kept = paddle.ones([x.shape[0]], dtype='bool')
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = paddle.concat([x[:1], x[1:] - x[:-1]])

        # save ketp for backward
        ctx.save_for_backward(kept)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensor()
        back = paddle.cumsum(kept.astype('int64'))
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class CamEncode(nn.Layer):
    def __init__(self, D, C, inputC):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.depthnet = nn.Conv2D(
            inputC, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return F.softmax(x, axis=1)

    def get_depth_feat(self, x):
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x, depth


class LiftSplatShoot(nn.Layer):
    def __init__(self,
                 lss=False,
                 final_dim=(900, 1600),
                 camera_depth_range=[4.0, 45.0, 1.0],
                 pc_range=[-50, -50, -5, 50, 50, 3],
                 downsample=4,
                 grid=3,
                 inputC=256,
                 camC=64):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            final_dim: actual RGB image size for actual BEV coordinates, default (900, 1600)
            downsample (int): the downsampling rate of the input camera feature spatial dimension (default (224, 400)) to final_dim (900, 1600), default 4.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            pc_range: point cloud range.
            inputC: input camera feature channel dimension (default 256).
            grid: stride for splat, see https://github.com/nv-tlabs/lift-splat-shoot.

        """
        super(LiftSplatShoot, self).__init__()
        self.pc_range = pc_range
        self.grid_conf = {
            'xbound': [pc_range[0], pc_range[3], grid],
            'ybound': [pc_range[1], pc_range[4], grid],
            'zbound': [pc_range[2], pc_range[5], grid],
            'dbound': camera_depth_range,
        }
        self.final_dim = final_dim
        self.grid = grid

        dx, bx, nx = gen_dx_bx(
            self.grid_conf['xbound'],
            self.grid_conf['ybound'],
            self.grid_conf['zbound'],
        )
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.register_buffer("dx", self.dx)
        self.register_buffer("bx", self.bx)
        # self.register_buffer("nx", self.nx)

        self.downsample = downsample
        self.fH, self.fW = self.final_dim[0] // self.downsample, self.final_dim[
            1] // self.downsample
        self.camC = camC
        self.inputC = inputC
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.inputC)
        self.register_buffer("frustum", self.frustum)

        # toggle using QuickCumsum vs. autograd
        # QuickCumsum is slower than autograd
        self.use_quickcumsum = False
        z = self.grid_conf['zbound']
        cz = int(self.camC * ((z[1] - z[0]) // z[2]))
        self.lss = lss
        self.bevencode = nn.Sequential(
            nn.Conv2D(cz, cz, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(cz), nn.ReLU(),
            nn.Conv2D(cz, 512, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(512), nn.ReLU(),
            nn.Conv2D(512, 512, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(512), nn.ReLU(),
            nn.Conv2D(512, inputC, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(inputC), nn.ReLU())
        if self.lss:
            self.bevencode = nn.Sequential(
                nn.Conv2D(cz, camC, kernel_size=1, padding=0, bias_attr=False),
                nn.BatchNorm2D(camC), BevEncode(inC=camC, outC=inputC))

        # defatul init to match torch
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Conv2D):
                reset_parameters(m)
            elif isinstance(m, nn.BatchNorm2D):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)

        self.apply(_init_weights)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = self.fH, self.fW
        ds = paddle.arange(
            *self.grid_conf['dbound'],
            dtype='float32').reshape([-1, 1, 1]).expand([-1, fH, fW])
        D, _, _ = ds.shape
        # yapf: disable
        xs = paddle.linspace(0, ogfW - 1, fW, dtype='float32').reshape([1, 1, fW]).expand([D, fH, fW])
        ys = paddle.linspace(0, ogfH - 1, fH, dtype='float32').reshape([1, fH, 1]).expand([D, fH, fW])
        # yapf: enable

        # D x H x W x 3
        frustum = paddle.stack([xs, ys, ds], axis=-1)
        return frustum

    def get_geometry(self, rots, trans, post_rots=None, post_trans=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        D, H, W = self.frustum.shape[:3]
        points = self.frustum.tile([B, N, 1, 1, 1, 1])  # B x N x D x H x W x 3

        # cam_to_lidar
        points = paddle.concat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:3]),
            axis=5)
        # TODO(liuxiao): this hacky way to make 7-dim tensor matmul fast
        points = points.reshape([B, N, D, H, W, 3, 1])
        points = points.reshape([-1, 3, 1])
        rots = rots.reshape([B, N, 1, 1, 1, 3,
                             3]).reshape([B * N, 1, 1, 1, 3, 3])
        rots = rots.expand([-1, D, H, W, -1, -1]).reshape([-1, 3, 3])

        points = paddle.matmul(rots, points).reshape([B, N, D, H, W, 3])
        points += trans.reshape([B, N, 1, 1, 1, 3])

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, H, W = x.shape

        x = x.reshape([B * N, C, H, W])
        x, depth = self.camencode(x)
        x = x.reshape([B, N, self.camC, self.D, H, W])
        x = x.transpose([0, 1, 3, 4, 5, 2])
        depth = depth.reshape([B, N, self.D, H, W])
        return x, depth

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape([Nprime, C])

        # flatten indices
        geom_feats = (
            (geom_feats - (self.bx - self.dx / 2.)) / self.dx).astype('int64')
        geom_feats = geom_feats.reshape([Nprime, 3])
        batch_ix = paddle.concat([
            paddle.full([Nprime // B, 1], ix, dtype='int64') for ix in range(B)
        ])
        geom_feats = paddle.concat((geom_feats, batch_ix), 1)
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        # import pdb;pdb.set_trace()
        x = x[kept]
        geom_feats = geom_feats[kept]
        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        # x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        x, geom_feats, ranks = paddle.index_select(
            x, sorts), paddle.index_select(geom_feats,
                                           sorts), paddle.index_select(
                                               ranks, sorts)
        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        # TODO(liuxiao): fix this complicate index assignment, the original assignament can't compute gradient when backward.
        final = paddle.zeros([B * self.nx[2] * self.nx[0] * self.nx[1], C])
        geom_feats = geom_feats[:, 3] * (self.nx[2] * self.nx[0] * self.nx[1]) \
                     + geom_feats[:, 2] * (self.nx[0] * self.nx[1]) \
                     + geom_feats[:, 0] * self.nx[1] \
                     + geom_feats[:, 1]

        # inplace version scatter is not supported in static mode
        if getattr(self, "export_model", False):
            final = paddle.scatter(final, geom_feats, x, overwrite=True)
        else:
            paddle.scatter_(final, geom_feats, x, overwrite=True)
        final = final.reshape([B, self.nx[2], self.nx[0], self.nx[1],
                               C]).transpose([0, 4, 1, 2, 3])

        return final

    def get_voxels(self,
                   x,
                   rots=None,
                   trans=None,
                   post_rots=None,
                   post_trans=None):
        geom = self.get_geometry(rots, trans, post_rots, post_trans)
        x, depth = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x)
        return x, depth

    def s2c(self, x):
        B, C, H, W, L = x.shape
        bev = paddle.reshape(x, [B, C * H, W, L])
        bev = bev.transpose([0, 1, 3, 2])
        return bev

    def forward(self,
                x,
                rots,
                trans,
                lidar2img_rt=None,
                bboxs=None,
                post_rots=None,
                post_trans=None,
                aug_bboxs=None,
                img_metas=None):
        x, depth = self.get_voxels(x, rots, trans, post_rots,
                                   post_trans)  # [B, C, H, W, L]
        bev = self.s2c(x)
        x = self.bevencode(bev)
        return x, depth
