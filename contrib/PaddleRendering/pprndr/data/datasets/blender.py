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
from pathlib import Path
from typing import List, Union

import imageio.v2 as imageio
import numpy as np

from pprndr.apis import manager
from pprndr.cameras import Cameras, CameraType
from pprndr.data.datasets.base import BaseDataset
from pprndr.data.transforms import Compose, TransformABC
from pprndr.geometries import SceneBox
from pprndr.utils import get_color

__all__ = ["BlenderDataset"]


@manager.DATASETS.add_component
class BlenderDataset(BaseDataset):
    """
    Args:
        dataset_root (str): Root directory of dataset.
        transforms (List[TransformABC], optional): Transforms to be applied on data. Defaults to None.
        camera_scale_factor (float, optional): How much to scale the camera origins by. Defaults to 1.0.
        background_color (Union[str, list, tuple], optional): Background color of the scene. Defaults to None.
        split (str, optional): Which split to use. Defaults to "train".
    """

    scene_bound = SceneBox(
        aabb=np.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=np.float32))

    def __init__(self,
                 dataset_root: str,
                 transforms: List[TransformABC] = None,
                 camera_scale_factor: float = 1.0,
                 background_color: Union[str, list, tuple] = None,
                 image_coords_offset: float = 0.5,
                 split: str = "train",
                 load_normals: bool = False):
        super(BlenderDataset, self).__init__()

        self.dataset_root = Path(dataset_root)
        self.transforms = Compose(transforms) if transforms else None
        self.camera_scale_factor = float(camera_scale_factor)
        self.image_coords_offset = image_coords_offset
        if background_color is not None:
            self.background_color = np.array(
                get_color(background_color), dtype=np.float32)
        else:
            self.background_color = None
        self._split = split.lower()
        self.load_normals = load_normals

    def _parse(self):
        with open(
                self.dataset_root / "transforms_{}.json".format(self.split),
                encoding="utf-8") as f:
            meta = json.load(f)

        self.image_paths = []
        self.normal_paths = []
        poses = []
        for frame in meta["frames"]:
            image_path = self.dataset_root / Path(frame["file_path"].replace(
                "./", "")).with_suffix(".png")
            self.image_paths.append(str(image_path))
            poses.append(frame["transform_matrix"])

            if self.load_normals:
                normal_path = self.dataset_root / Path(
                    (frame["file_path"] + "_normal").replace(
                        "./", "")).with_suffix(".png")
                self.normal_paths.append(str(normal_path))
        poses = np.array(poses, dtype=np.float32)

        image_height, image_width = imageio.imread(
            self.image_paths[0]).shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / math.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        c2w_matrices = poses[:, :3]  # [N, 3, 4]
        c2w_matrices[..., 3] *= self.camera_scale_factor

        self._cameras = Cameras(
            c2w_matrices=c2w_matrices,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            image_height=image_height,
            image_width=image_width,
            camera_type=CameraType.PERSPECTIVE)

    @property
    def cameras(self) -> Cameras:
        return self._cameras

    @property
    def split(self):
        return self._split

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample = {
            "camera_id": idx,
            "image": self.image_paths[idx],
        }
        if self.background_color is not None:
            sample["background_color"] = self.background_color

        if self.load_normals:
            sample["normal"] = self.normal_paths[idx]
        if self.transforms:
            sample = self.transforms(sample)

        return sample
