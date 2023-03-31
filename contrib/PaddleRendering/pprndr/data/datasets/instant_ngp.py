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

import json
import math
import os.path as osp
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from pprndr.apis import manager
from pprndr.cameras import Cameras, CameraType
from pprndr.cameras.camera_functionals import get_distortion_coeffs
from pprndr.data.datasets.base import BaseDataset
from pprndr.data.transforms import Compose, TransformABC
from pprndr.geometries import SceneBox
from pprndr.utils import get_color
from pprndr.utils.logger import logger

__all__ = ["InstantNGPDataset"]


@manager.DATASETS.add_component
class InstantNGPDataset(BaseDataset):
    """
    Args:
        dataset_root (str): Root directory of dataset.
        transforms (List[TransformABC], optional): Transforms to be applied on data. Defaults to None.
        scene_scale_factor (float, optional): How much to scale the scene. Defaults to .33.
        background_color (Union[str, list, tuple], optional): Background color of the scene. Defaults to None.
        split (str, optional): Which split to use. Defaults to "train".
    """

    def __init__(self,
                 dataset_root: str,
                 transforms: List[TransformABC] = None,
                 scene_scale_factor: float = .33,
                 background_color: Union[str, list, tuple] = None,
                 split: str = "train"):
        super(InstantNGPDataset, self).__init__()

        self.dataset_root = Path(dataset_root)
        self.transforms = Compose(transforms) if transforms else None
        self.scene_scale_factor = float(scene_scale_factor)
        if background_color is not None:
            self.background_color = np.array(
                get_color(background_color), dtype=np.float32)
        else:
            self.background_color = None
        self._split = split.lower()

    def _parse(self):
        with open(self.dataset_root / "transforms.json", encoding="utf-8") as f:
            meta = json.load(f)

        self.image_paths = []
        poses = []
        num_skipped_paths = 0

        if self.split == "train":
            frames = meta["frames"][1:]
        elif self.split == "val":
            frames = meta["frames"][:1]
        else:
            frames = meta["frames"]

        for frame in frames:
            image_path = self.dataset_root / Path(frame["file_path"])
            if osp.exists(image_path):
                self.image_paths.append(str(image_path))
                poses.append(frame["transform_matrix"])
            else:
                num_skipped_paths += 1

        if num_skipped_paths > 0:
            logger.info(
                f"Skipped {num_skipped_paths} non-existent paths in {self.dataset_root / 'transforms.json'}."
            )

        assert len(self.image_paths) > 0, "No valid image paths found."

        poses = np.array(poses, dtype=np.float32)
        poses[:, :3, 3] *= self.scene_scale_factor
        c2w_matrices = poses[:, :3]

        distortion_coeffs = get_distortion_coeffs(
            k1=meta["k1"], k2=meta["k2"], p1=meta["p1"], p2=meta["p2"])

        aabb_scale = meta["aabb_scale"]
        self.scene_box = SceneBox(
            aabb=np.array([[-aabb_scale, -aabb_scale, -aabb_scale],
                           [aabb_scale, aabb_scale, aabb_scale]],
                          dtype=np.float32))
        fx, fy = InstantNGPDataset.get_focal_lengths(meta)

        self._cameras = Cameras(
            c2w_matrices=c2w_matrices,
            fx=fx,
            fy=fy,
            cx=meta["cx"],
            cy=meta["cy"],
            distortion_coeffs=distortion_coeffs,
            image_height=meta["h"],
            image_width=meta["w"],
            camera_type=CameraType.PERSPECTIVE)

    @property
    def cameras(self) -> Cameras:
        return self._cameras

    @property
    def split(self):
        return self._split

    @staticmethod
    def get_focal_lengths(meta: Dict) -> Tuple:
        fx, fy = 0, 0

        def fov_to_focal_length(rad, res):
            return 0.5 * res / math.tan(0.5 * rad)

        if "fl_x" in meta:
            fx = meta["fl_x"]
        elif "x_fov" in meta:
            fx = fov_to_focal_length(math.radians(meta["x_fov"]), meta["w"])
        elif "camera_angle_x" in meta:
            fx = fov_to_focal_length(meta["camera_angle_x"], meta["w"])

        if "fl_y" in meta:
            fy = meta["fl_y"]
        elif "y_fov" in meta:
            fy = fov_to_focal_length(math.radians(meta["y_fov"]), meta["h"])
        elif "camera_angle_y" in meta:
            fy = fov_to_focal_length(meta["camera_angle_y"], meta["h"])

        if fx == 0 or fy == 0:
            raise AttributeError(
                "Focal length cannot be calculated from transforms.json (missing fields)."
            )

        return fx, fy

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
