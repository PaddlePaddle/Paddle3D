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

import os
import sys
import json
from typing import List, Union, Optional

from pathlib import Path
import imageio.v2 as imageio
import cv2
from glob import glob
import numpy as np

from pprndr.apis import manager
from pprndr.cameras import Cameras, CameraType
from pprndr.data.datasets.base import BaseDataset
from pprndr.data.transforms import Compose, TransformABC
from pprndr.geometries import SceneBox
from pprndr.utils import get_color

__all__ = ["LLFFDataset"]


@manager.DATASETS.add_component
class LLFFDataset(BaseDataset):
    """
    Args:
        dataset_root (str): Root directory of dataset.
        transforms (List[TransformABC], optional): Transforms to be applied on data. Defaults to None.
        camera_scale_factor (float, optional): How much to scale the camera origins by. Defaults to 1.0.
        background_color (Union[str, list, tuple], optional): Background color of the scene. Defaults to None.
        split (str, optional): Which split to use. Defaults to "train".
    """

    def __init__(self,
                 dataset_root: str,
                 render_cameras_name,
                 object_cameras_name,
                 transforms: List[TransformABC] = None,
                 camera_scale_factor: float = 1.0,
                 camera_axis_convention: str = "OpenCV",
                 image_coords_offset: float = 0.0,
                 background_color: Union[str, list, tuple] = None,
                 validate_mesh: Optional[str] = "neus_style",
                 mesh_resolution: Optional[str] = 64,
                 eval_to_cpu: Optional[bool] = False,
                 world_space_for_mesh: Optional[bool] = False,
                 bound_min: Optional[list] = None,
                 bound_max: Optional[list] = None,
                 split: str = "train"):
        super(LLFFDataset, self).__init__()
        self.data_dir = dataset_root
        self.camera_axis_convention = camera_axis_convention
        self.render_cameras_name = render_cameras_name
        self.object_cameras_name = object_cameras_name
        self.validate_mesh = validate_mesh
        self.mesh_resolution = mesh_resolution
        self.eval_to_cpu = eval_to_cpu
        self.world_space_for_mesh = world_space_for_mesh
        self.bound_min = bound_min
        self.bound_max = bound_max

        self.transforms = Compose(transforms) if transforms else None
        self.camera_scale_factor = float(camera_scale_factor)
        self.image_coords_offset = float(image_coords_offset)
        if background_color is not None:
            self.background_color = np.array(
                get_color(background_color), dtype=np.float32)
        else:
            self.background_color = None

        self._split = split.lower()

    @property
    def split(self):
        return self._split

    def load_K_Rt_from_P(self, filename, P=None):
        """
        	load_K_Rt_from_P
        """
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]]
                     for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()

        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        return intrinsics, pose

    def _parse(self):
        # Get camera metas
        camera_dict = np.load(
            os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict

        # Get image list, image num.
        self.image_paths = sorted(
            glob(os.path.join(self.data_dir, 'image/*.png')))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(
                glob(os.path.join(self.data_dir, 'image/*.jpg')))
        self.n_images = len(self.image_paths)

        # Get mask list
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        if len(self.masks_lis) == 0:
            self.masks_lis = sorted(
                glob(os.path.join(self.data_dir, 'mask/*.jpg')))

        # Load poses.
        # world_mat is a projection matrix from world to image
        world_mats_np = [
            camera_dict['world_mat_%d' % idx].astype(np.float32)
            for idx in range(self.n_images)
        ]

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        scale_mats_np = [
            camera_dict['scale_mat_%d' % idx].astype(np.float32)
            for idx in range(self.n_images)
        ]

        intrinsics_all = []
        pose_all = []

        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = self.load_K_Rt_from_P(None, P)
            intrinsics_all.append(intrinsics)
            pose_all.append(pose)

        intrinsics_all = np.stack(intrinsics_all)
        fx = intrinsics_all[:, 0, 0]
        fy = intrinsics_all[:, 1, 1]
        cx = intrinsics_all[:, 0, 2]
        cy = intrinsics_all[:, 1, 2]

        c2w_matrices = np.stack(pose_all)[:, :3, :]  # [n_images, 4, 4]
        c2w_matrices[..., 3] *= self.camera_scale_factor

        tmp_img = cv2.imread(self.image_paths[0])
        image_height = tmp_img.shape[0]
        image_width = tmp_img.shape[1]

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])

        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(
            os.path.join(self.data_dir,
                         self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(
            scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(
            scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]

        object_bbox_min = object_bbox_min[:3, 0]
        object_bbox_max = object_bbox_max[:3, 0]

        self._object_bbox_min = object_bbox_min
        self._object_bbox_max = object_bbox_max

        self._cameras = Cameras(
            c2w_matrices=c2w_matrices,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            axis_convention=self.camera_axis_convention,
            image_height=image_height,
            image_width=image_width,
            camera_type=CameraType.PERSPECTIVE)

    @property
    def cameras(self) -> Cameras:
        return self._cameras

    @property
    def object_bbox_min(self):
        return self._object_bbox_min

    @property
    def object_bbox_max(self):
        return self._object_bbox_max

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample = {
            "camera_id": idx,
            "image": self.image_paths[idx],
        }
        if self.background_color is not None:
            sample["background_color"] = self.background_color

        if self.transforms:
            sample = self.transforms(sample)

        return sample
