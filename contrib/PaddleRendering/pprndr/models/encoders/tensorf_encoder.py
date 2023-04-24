#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
""" The code is mainly based on: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/field_components/embedding.py """

from typing_extensions import Literal
import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pprndr.apis import manager

__all__ = ["TensorCPEncoder", "TensorVMEncoder", "TriplaneEncoder"]


@manager.ENCODERS.add_component
class TensorCPEncoder(nn.Layer):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self,
                 resolution: int = 256,
                 num_components: int = 24,
                 init_scale: float = 0.1) -> None:
        super(TensorCPEncoder, self).__init__()

        self.resolution = resolution
        self.num_components = num_components

        weight_attr = paddle.ParamAttr(learning_rate=20)
        line_coef_init = nn.initializer.Assign(
            init_scale * paddle.randn([3, num_components, resolution, 1]))
        self.line_coef = paddle.create_parameter(shape=[3, num_components, resolution, 1], dtype='float32',  \
                                                 default_initializer=line_coef_init, attr=weight_attr)

    @property
    def output_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: paddle.Tensor):

        line_coord = paddle.stack(
            [in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]],
            axis=0)  # [3, ...]
        line_coord = paddle.stack([paddle.zeros_like(line_coord), line_coord],
                                  axis=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        line_coord = line_coord.detach().reshape([3, -1, 1, 2])

        line_features = F.grid_sample(
            self.line_coef, line_coord,
            align_corners=True)  # [3, Components, -1, 1]
        features = paddle.prod(line_features, axis=0)
        features = paddle.moveaxis(
            features.reshape([self.num_components, *in_tensor.shape[:-1]]), 0,
            -1)
        return features  # [..., Components]

    @paddle.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution: Target resolution.
        """

        paddle.assign(
            F.interpolate(
                self.line_coef,
                size=(resolution, 1),
                mode="bilinear",
                align_corners=True), self.line_coef)

        self.resolution = resolution


@manager.ENCODERS.add_component
class TensorVMEncoder(nn.Layer):
    """Learned vector-matrix encoding proposed by TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(
            self,
            resolution: int = 128,
            num_components: int = 24,
            init_scale: float = 0.1,
    ) -> None:
        super(TensorVMEncoder, self).__init__()

        self.resolution = resolution
        self.num_components = num_components

        weight_attr = paddle.ParamAttr(learning_rate=20)

        plane_init = nn.initializer.Assign(init_scale * paddle.randn(
            [3, num_components, resolution, resolution]))
        self.plane_coef = paddle.create_parameter(shape=[3, num_components, resolution, resolution], dtype='float32',  \
                                                 default_initializer=plane_init, attr=weight_attr)

        line_init = nn.initializer.Assign(
            init_scale * paddle.randn([3, num_components, resolution, 1]))
        self.line_coef = paddle.create_parameter(shape=[3, num_components, resolution, resolution], dtype='float32',  \
                                                 default_initializer=line_init, attr=weight_attr)

    @property
    def output_dim(self) -> int:
        return self.num_components * 3

    def forward(self, in_tensor: paddle.Tensor):
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1],

        Returns: Encoded position
        """

        plane_coord = paddle.stack((in_tensor.index_select(paddle.to_tensor([0, 1]), axis=-1), \
                                         in_tensor.index_select(paddle.to_tensor([0, 2]), axis=-1), \
                                         in_tensor.index_select(paddle.to_tensor([1, 2]), axis=-1)), axis=0)
        line_coord = paddle.stack(
            [in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]],
            axis=0)  # [3, ...]
        line_coord = paddle.stack([paddle.zeros_like(line_coord), line_coord],
                                  axis=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().reshape([3, -1, 1, 2])
        line_coord = line_coord.detach().reshape([3, -1, 1, 2])

        plane_features = F.grid_sample(
            self.plane_coef, plane_coord,
            align_corners=True)  # [3, Components, -1, 1]
        line_features = F.grid_sample(
            self.line_coef, line_coord,
            align_corners=True)  # [3, Components, -1, 1]

        features = plane_features * line_features  # [3, Components, -1, 1]
        features = paddle.moveaxis(
            features.reshape([3 * self.num_components, *in_tensor.shape[:-1]]),
            0, -1)
        return features  # [..., 3 * Components]

    @paddle.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """

        paddle.assign(
            F.interpolate(
                self.plane_coef,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=True), self.plane_coef)

        paddle.assign(
            F.interpolate(
                self.line_coef,
                size=(resolution, 1),
                mode="bilinear",
                align_corners=True), self.line_coef)

        self.resolution = resolution


@manager.ENCODERS.add_component
class TriplaneEncoder(nn.Layer):
    """Learned triplane encoding

    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the
    three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].

    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing
    and symmetrical, unlike with VM decomposition where we needed one component with a vector along all the x, y, z
    directions for symmetry.

    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting
    at the origin, and the encoding being the element-wise product of the element at the projection of [i, j, k] on
    these planes.

    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of grid.
        num_components: The number of scalar triplanes to use (ie: output feature size)
        init_scale: The scale of the initial values of the planes
        product: Whether to use the element-wise product of the planes or the sum
    """

    def __init__(
            self,
            resolution: int = 32,
            num_components: int = 64,
            init_scale: float = 0.1,
            reduce: Literal["sum", "product"] = "sum",
    ) -> None:
        super(TriplaneEncoder, self).__init__()

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce

        weight_attr = paddle.ParamAttr(learning_rate=20)
        plane_init = nn.initializer.Assign(init_scale * paddle.randn(
            [3, num_components, resolution, resolution]))
        self.plane_coef = paddle.create_parameter(shape=[3, num_components, resolution, resolution], dtype='float32',  \
                                                 default_initializer=plane_init, attr=weight_attr)

    @property
    def output_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: paddle.Tensor):
        """Sample features from this encoder. Expects in_tensor to be in range [0, resolution]"""

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape([-1, 3])

        plane_coord = paddle.stack((in_tensor.index_select(paddle.to_tensor([0, 1]), axis=-1), \
                                         in_tensor.index_select(paddle.to_tensor([0, 2]), axis=-1), \
                                         in_tensor.index_select(paddle.to_tensor([1, 2]), axis=-1)), axis=0)

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().reshape([3, -1, 1, 2])
        plane_features = F.grid_sample(
            self.plane_coef, plane_coord,
            align_corners=True)  # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(0).squeeze(
                -1).T  # [flattened_bs, num_components]
        else:
            plane_features = plane_features.sum(0).squeeze(-1).T

        return plane_features.reshape(
            [*original_shape[:-1], self.num_components])

    @paddle.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """

        paddle.assign(
            F.interpolate(
                self.plane_coef,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=True), self.plane_coef)

        self.resolution = resolution
