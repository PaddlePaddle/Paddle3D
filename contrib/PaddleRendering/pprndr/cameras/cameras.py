# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Union

import numpy as np
import paddle
import paddle.nn.functional as F

from pprndr.cameras.camera_functionals import radial_n_tangential_undistort
from pprndr.cameras.rays import RayBundle
from pprndr.utils.logger import logger

__all__ = ["CameraType", "Cameras"]


class CameraType(IntEnum):
    """Camera type enum"""
    PERSPECTIVE = 0
    FISHEYE = 1


@dataclass
class Cameras:
    c2w_matrices: Union[np.ndarray, paddle.Tensor]
    fx: Union[np.ndarray, paddle.Tensor, float]
    fy: Union[np.ndarray, paddle.Tensor, float]
    cx: Union[np.ndarray, paddle.Tensor, float]
    cy: Union[np.ndarray, paddle.Tensor, float]
    image_height: Union[np.ndarray, paddle.Tensor, int]
    image_width: Union[np.ndarray, paddle.Tensor, int]
    axis_convention: str = "OpenGL"
    distortion_coeffs: Optional[Union[np.ndarray, paddle.Tensor]] = None
    camera_type: Optional[
        Union[np.ndarray, paddle.Tensor, int, List[CameraType],
              CameraType]] = CameraType.PERSPECTIVE
    place: Optional[Union[str, paddle.CPUPlace, paddle.CUDAPlace, paddle.
                          CUDAPinnedPlace]] = "cpu"

    def __init__(
            self,
            c2w_matrices: Union[np.ndarray, paddle.Tensor],
            fx: Union[np.ndarray, paddle.Tensor, float],
            fy: Union[np.ndarray, paddle.Tensor, float],
            cx: Union[np.ndarray, paddle.Tensor, float],
            cy: Union[np.ndarray, paddle.Tensor, float],
            image_height: Union[np.ndarray, paddle.Tensor, int],
            image_width: Union[np.ndarray, paddle.Tensor, int],
            axis_convention: Optional[str] = "OpenGL",
            distortion_coeffs: Optional[
                Union[np.ndarray, paddle.Tensor]] = None,
            camera_type: Optional[
                Union[np.ndarray, paddle.Tensor, int, List[CameraType],
                      CameraType]] = CameraType.PERSPECTIVE,
            place: Optional[Union[str, paddle.CPUPlace, paddle.
                                  CUDAPlace, paddle.CUDAPinnedPlace]] = "cpu"):
        self._set_place(place)

        self.axis_convention = axis_convention
        if c2w_matrices.ndim == 2:
            c2w_matrices = c2w_matrices.unsqueeze_(0)
            self._num_cameras = 1
        else:
            self._num_cameras = c2w_matrices.shape[0]

        # extrinsics
        self.c2w_matrices = paddle.to_tensor(
            c2w_matrices, place=self.place)  # (N, 3, 4)

        # intrinsics
        self.fx = paddle.to_tensor(
            fx, dtype="float32",
            place=self.place).broadcast_to([self._num_cameras])
        self.fy = paddle.to_tensor(
            fy, dtype="float32",
            place=self.place).broadcast_to([self._num_cameras])
        self.cx = paddle.to_tensor(
            cx, dtype="float32",
            place=self.place).broadcast_to([self._num_cameras])
        self.cy = paddle.to_tensor(
            cy, dtype="float32",
            place=self.place).broadcast_to([self._num_cameras])

        # heights, widths
        self._image_height = paddle.to_tensor(
            image_height, dtype="int64",
            place=self.place).broadcast_to([self._num_cameras])
        self._image_width = paddle.to_tensor(
            image_width, dtype="int64",
            place=self.place).broadcast_to([self._num_cameras])

        # distortions
        if distortion_coeffs is not None:
            self.distortion_coeffs = paddle.to_tensor(
                distortion_coeffs, dtype="float32",
                place=self.place).broadcast_to([self._num_cameras, 6])
        else:
            self.distortion_coeffs = None

        # camera types
        self.camera_types = paddle.to_tensor(
            camera_type, place=self.place).broadcast_to([self._num_cameras])

    def _set_place(self, place: Union[str, paddle.CPUPlace, paddle.
                                      CUDAPlace, paddle.CUDAPinnedPlace]):
        if isinstance(place, str):
            place = place.lower()
            if place == "cpu":
                self.place = paddle.CPUPlace()
            elif place == "cuda":
                self.place = None
            elif place == "cuda_pinned":
                self.place = paddle.CUDAPinnedPlace()
            else:
                raise ValueError("Invalid place: {}".format(place))
        else:
            self.place = place

    def cuda(self) -> "Cameras":
        """
        Copy to GPU.
        """
        return Cameras(
            c2w_matrices=self.c2w_matrices,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            image_height=self._image_height,
            image_width=self._image_width,
            distortion_coeffs=self.distortion_coeffs,
            camera_type=self.camera_types,
            axis_convention=self.axis_convention,
            place="cuda")

    def __getitem__(self, indices) -> "Cameras":
        return Cameras(
            c2w_matrices=self.c2w_matrices[indices],
            fx=self.fx[indices],
            fy=self.fy[indices],
            cx=self.cx[indices],
            cy=self.cy[indices],
            image_height=self._image_height[indices],
            image_width=self._image_width[indices],
            distortion_coeffs=self.distortion_coeffs[indices]
            if self.distortion_coeffs is not None else None,
            camera_type=self.camera_types[indices])

    def __len__(self):
        return self._num_cameras

    def get_image_coords(self, offset: float = .5,
                         step: int = 1) -> paddle.Tensor:
        """
        Generate coordinates on image.

        Args:
            offset: Offset wrt. the upper left corner of each pixel grid. Default: 0.5 (the center of each pixel).

        Returns:
            Tensor of image coordinates, (H, W, 2).
        """

        row, col = paddle.meshgrid(
            paddle.arange(self._image_height[0], step=step),
            paddle.arange(self._image_width[0], step=step))
        image_coords = paddle.stack([row, col],
                                    axis=-1).astype("float32") + offset

        return image_coords

    def convert_axis(self, target: str):
        if self.axis_convention not in ["OpenCV", "OpenGL"
                                        ] or target not in ["OpenGL"]:
            raise ValueError("Axis convention out of scope.")
        else:
            if self.axis_convention == target:
                logger.info("Axis convertion is not needed.")
                R_axis = paddle.to_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                          dtype=paddle.float32)
                return R_axis
            else:
                if self.axis_convention == "OpenCV" and target == "OpenGL":
                    R_axis = paddle.to_tensor(
                        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                        dtype=paddle.float32)
                    return R_axis
                else:
                    raise NotImplementedError(
                        "The specified axis convertion is not implemented.")

    def generate_rays(
            self,
            image_coords: Optional[paddle.Tensor] = None,
            camera_ids: Union[paddle.Tensor, int, None] = None) -> RayBundle:
        """
        Generate rays from camera specified by camera_ids.

        This method accepts the following combinations of `image_coords` and `camera_ids`:
            Firstly, if `image_coords` is None, coordinates of the entire image are used. Then:
            1. `image_coords` is a tensor of shape (N, 2) and `camera_ids` is None.
               - All `image_coords` are from camera 0 (only used when num_cameras is 1).
            2. `image_coords` is a tensor of shape (N, 2) and `camera_ids` is an integer.
                - All `image_coords` are from camera `camera_ids`.
            3. `image_coords` is a tensor of shape (N, 2) and `camera_ids` is a tensor of shape (B,).
                - All B cameras have the same `image_coords`.
            4. `image_coords` is a tensor of shape (N, 2) and `camera_ids` is a tensor of shape (N,).
                - `image_coords` and `camera_ids` have a one-to-one correspondence..

        Args:
            image_coords: Tensor of image coordinates.
            camera_ids: Tensor of camera ids or an integer representing a single camera id.
        """
        if image_coords is None:
            image_coords = self.get_image_coords().reshape([-1, 2])

        if camera_ids is None:
            assert self._num_cameras == 1, "`camera_ids` must be specified when there are multiple cameras."
            camera_ids = paddle.zeros([len(image_coords), 1], dtype="int32")
        elif isinstance(camera_ids, int):
            camera_ids = paddle.full([len(image_coords), 1],
                                     camera_ids,
                                     dtype="int32")
        elif len(image_coords) != len(camera_ids):
            num_cameras = len(camera_ids)
            camera_ids = camera_ids.repeat_interleave(len(image_coords), axis=0)
            if num_cameras > 1:
                image_coords = image_coords.tile([num_cameras, 1])

        if camera_ids.ndim == 1:
            camera_ids = camera_ids.unsqueeze(-1)

        fx = paddle.index_select(self.fx, camera_ids, axis=0)
        fy = paddle.index_select(self.fy, camera_ids, axis=0)
        cx = paddle.index_select(self.cx, camera_ids, axis=0)
        cy = paddle.index_select(self.cy, camera_ids, axis=0)

        v = image_coords[:, 0]
        u = image_coords[:, 1]
        x_coords = (u - cx) / fx  # (N,)
        y_coords = -(v - cy) / fy  # (N,)
        x_offsets = (u - cx + 1.) / fx  # (N,)
        y_offsets = -(v - cy + 1.) / fy  # (N,)

        coord_stack = paddle.stack([
            paddle.stack([x_coords, y_coords], axis=-1),
            paddle.stack([x_offsets, y_coords], axis=-1),
            paddle.stack([x_coords, y_offsets], axis=-1),
        ],
                                   axis=0)  # (3, N, 2)

        if self.distortion_coeffs is not None:
            distortion_coeffs = paddle.index_select(
                self.distortion_coeffs, camera_ids, axis=0)
            coord_stack = radial_n_tangential_undistort(coord_stack,
                                                        distortion_coeffs)

        if self.camera_types[0].item() == CameraType.PERSPECTIVE.value:
            directions_stack = paddle.concat(
                [coord_stack,
                 paddle.full([3, coord_stack.shape[1], 1], -1.0)],
                axis=-1)  # (3, N, 3)
        elif self.camera_types[0].item() == CameraType.FISHEYE.value:
            theta = paddle.clip(
                paddle.sqrt(paddle.sum(coord_stack**2, axis=-1)),
                min=0.,
                max=math.pi)  # (3, N)
            sin_theta = paddle.sin(theta)
            cos_theta = paddle.cos(theta)
            directions_stack = paddle.concat(
                [coord_stack * sin_theta / theta, -cos_theta.unsqueeze_(-1)],
                axis=-1)  # (3, N, 3)
        else:
            raise ValueError("Unsupported camera type: {}".format(
                self.camera_types[0]))

        c2w_matrices = paddle.index_select(
            self.c2w_matrices, camera_ids, axis=0)

        rotation = c2w_matrices[:, :3, :3]  # (N, 3, 3)

        if self.axis_convention != "OpenGL":
            trans_mat = self.convert_axis(target="OpenGL")
            trans_mat = paddle.tile(trans_mat[None, :, :],
                                    (rotation.shape[0], 1, 1))
            # Convert rotation into OpenGL concention
            rotation = paddle.matmul(rotation, trans_mat)

        directions_stack = paddle.sum(
            directions_stack.unsqueeze(-2) * rotation, axis=-1)

        directions_stack = F.normalize(directions_stack, p=2, axis=-1)

        directions = directions_stack[0]  # (N, 3)
        dx = paddle.sqrt(
            paddle.sum(
                (directions - directions_stack[1])**2, axis=-1,
                keepdim=True))  # (N, 1)
        dy = paddle.sqrt(
            paddle.sum(
                (directions - directions_stack[2])**2, axis=-1,
                keepdim=True))  # (N, 1)
        pixel_area = dx * dy  # (N, 1)

        # Get origins
        origins = c2w_matrices[..., :3, 3]  # (N, 3)

        return RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_ids=camera_ids)
