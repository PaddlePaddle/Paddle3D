#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License")
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Dict, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from pprndr.apis import manager
from pprndr.cameras.rays import RaySamples

try:
    import trunc_exp
except ModuleNotFoundError:
    from pprndr.cpp_extensions import trunc_exp

from pprndr.models.fields import BaseField, NeRFField

__all__ = ["NeuSField", "NeRFPPField"]


@manager.FIELDS.add_component
class NeRFPPField(NeRFField):
    """
    NeRF++ Field according to the paper.
    """

    def __init__(
            self,
            dir_encoder: nn.Layer,
            pos_encoder: nn.Layer,
            stem_net: nn.Layer,
            density_head: nn.Layer,
            color_head: nn.Layer,
            density_noise: float = None,
            density_bias: float = None,
            rgb_padding: float = None,
    ):
        super(NeRFPPField, self).__init__(
            dir_encoder=dir_encoder,
            pos_encoder=pos_encoder,
            stem_net=stem_net,
            density_head=density_head,
            color_head=color_head,
            density_noise=density_noise,
            density_bias=density_bias,
            rgb_padding=rgb_padding)


@manager.FIELDS.add_component
class NeuSField(BaseField):
    """
    SDF and a rendring net
    """

    def __init__(self, pos_encoder: nn.Layer, view_encoder: nn.Layer,
                 sdf_network: nn.Layer, color_network: nn.Layer,
                 variance_init_val: float, scale: float):
        super(NeuSField, self).__init__()

        self.scale = scale

        # Encoder
        self.pos_encoder = pos_encoder
        self.view_encoder = view_encoder

        # Networks
        self.sdf_network = sdf_network
        self.color_network = color_network
        self.deviation_network = SingleVarianceNetwork(variance_init_val)

    def get_sdf_from_pts(self, pts: paddle.Tensor):
        # pts : (1, N, 3)
        pts = self.pos_encoder(pts)
        outputs = self.sdf_network(pts)
        signed_distances = outputs[:, :, :1]
        return signed_distances

    def get_sdf_output(
            self,
            ray_samples: RaySamples,
            which_pts: str = "mid_points"  # or bin_points
    ):
        if which_pts == "mid_points":
            pts = ray_samples.frustums.positions
        else:
            assert (which_pts == "bin_points")
            pts = ray_samples.frustums.bin_points

        pts = pts * self.scale
        pts = self.pos_encoder(pts)
        outputs = self.sdf_network(pts)
        signed_distances = outputs[:, :, :1] / self.scale
        sdf_features = outputs[:, :, 1:]
        return signed_distances, sdf_features

    def get_gradients(self,
                      ray_samples: RaySamples,
                      which_pts: str = "mid_points") -> paddle.Tensor:
        if which_pts == "mid_points":
            pts = ray_samples.frustums.positions
        else:
            assert (which_pts == "bin_points")
            pts = ray_samples.frustums.bin_points
        pts = pts * self.scale
        pts.stop_gradient = False
        pts_enc = self.pos_encoder(pts)
        signed_distances = self.sdf_network(pts_enc)[:, :, :1] / self.scale
        d_outputs = paddle.ones_like(signed_distances)  # How to set it no_grad?
        d_outputs.stop_gradient = False
        gradients = paddle.grad(
            outputs=signed_distances,
            inputs=pts,
            grad_outputs=d_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        return gradients

    def get_colors(self,
                   ray_samples: RaySamples,
                   gradients: paddle.Tensor,
                   sdf_features: paddle.Tensor,
                   which_pts: str = "mid_points") -> paddle.Tensor:
        # Get inputs
        if which_pts == "mid_points":
            pts = ray_samples.frustums.positions
        else:
            assert (which_pts == "bin_points")
            pts = ray_samples.frustums.bin_points

        dirs = ray_samples.frustums.directions

        # Encode view_dirs
        dirs = self.view_encoder(dirs)  # (view[B, 3] & encoded(view)[B, 3 * 4])
        rendering_input = paddle.concat([pts, dirs, gradients, sdf_features],
                                        axis=-1)
        color = self.color_network(rendering_input)

        return color

    def get_inside_filter_by_norm(self,
                                  ray_samples: RaySamples,
                                  which_pts: str = "mid_points"
                                  ) -> paddle.Tensor:
        if which_pts == "mid_points":
            pts = ray_samples.frustums.positions
        else:
            assert (which_pts == "bin_points")
            pts = ray_samples.frustums.bin_points
        batch_size, n_samples, _ = pts.shape
        pts_norm = paddle.linalg.norm(pts, p=2, axis=-1, keepdim=True)
        inside_sphere = (pts_norm < 1.0).astype('float32').detach()
        relax_inside_sphere = (pts_norm < 1.2).astype('float32').detach()
        return inside_sphere, relax_inside_sphere

    def get_inv_s(self, batch_size: int, n_samples: int):
        inv_s = self.deviation_network(paddle.zeros([1, 3]))[:, :1].clip(
            1e-6, 1e6)
        inv_s = inv_s.unsqueeze(axis=-1)
        inv_s = paddle.repeat_interleave(inv_s, batch_size, axis=0)
        inv_s = paddle.repeat_interleave(inv_s, n_samples, axis=1)
        return inv_s


class SingleVarianceNetwork(nn.Layer):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        variance = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Assign(
                np.array([init_val])))
        self.add_parameter('variance', variance)

    def forward(self, x):
        return paddle.ones([len(x), 1]) * paddle.exp(self.variance * 10.0)
