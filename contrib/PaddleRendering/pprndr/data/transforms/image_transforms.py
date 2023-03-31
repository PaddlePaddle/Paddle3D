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

import os.path as osp
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from pprndr.apis import manager
from pprndr.data.transforms.base import TransformABC
from pprndr.utils.logger import logger

__all__ = ["LoadImage", "AlphaBlending", "Normalize", "LowPass"]


@manager.TRANSFORMS.add_component
class LoadImage(TransformABC):
    """
    Load image from file path.
    """
    backends = {"pillow", "cv2"}
    var_channel_suffixes = {
        ".exr", ".png", ".PNG"
    }  # image formats that may contain multiple channels
    fixed_channel_suffixes = {".jpg", ".jpeg", ".JPG", ".JPEG"
                              }  # image formats that contain only RGB channels

    def __init__(self, backend="pillow", cv2_flag=None):
        """
        Args:
            backend (str): backend to load image, default is "pillow".
            cv2_flag (int): flag to load image (if backend is "cv2"), default is None.
        """
        backend = backend.lower()
        if backend not in self.backends:
            raise ValueError(
                f"backend {backend} is not supported, only support {self.backends}"
            )
        self.backend = backend
        if self.backend == "pillow" and cv2_flag is not None:
            logger.warning(
                f"Loading image with Pillow backend, 'cv2_flag={cv2_flag}' will be ignored."
            )
            self.cv2_flag = None
        else:
            self.cv2_flag = cv2_flag

    @staticmethod
    def pillow_reader(image_path):
        return np.asarray(
            Image.open(image_path), dtype=np.uint8).astype(
                np.float32, copy=False)

    @staticmethod
    def cv2_reader(image_path, cv2_flag):
        suffix = osp.splitext(image_path)[1]
        if cv2_flag is None:
            if suffix in LoadImage.var_channel_suffixes:
                cv2_flag = cv2.IMREAD_UNCHANGED
            elif suffix in LoadImage.fixed_channel_suffixes:
                cv2_flag = cv2.IMREAD_COLOR
            else:
                raise ValueError(f"Unsupported image format {suffix}")

        image = cv2.imread(image_path, cv2_flag)

        num_channles = image.shape[-1]
        if num_channles == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif num_channles == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            pass

        return image

    def __call__(self, sample: Dict) -> Dict:
        """
        """
        # TODO(will-jl944): Simplify reader picking logic
        if self.backend == "pillow":
            sample["image"] = LoadImage.pillow_reader(sample["image"])
        else:
            sample["image"] = LoadImage.cv2_reader(sample["image"],
                                                   self.cv2_flag)

        return sample


@manager.TRANSFORMS.add_component
class AlphaBlending(TransformABC):
    """
    Blend image and background.
    """

    def __call__(self, sample: Dict) -> Dict:
        """
        """
        image = sample["image"]
        background_color = sample.get("background_color", None)

        if background_color is not None and image.shape[-1] == 4:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + background_color * (1. - alpha)
        else:
            image = image[:, :, :3]

        sample["image"] = image

        return sample


@manager.TRANSFORMS.add_component
class Normalize(TransformABC):
    """
    Normalize image to [0, 1] (by dividing by 255.0).
    """

    def __call__(self, sample: Dict) -> Dict:
        """
        """
        sample["image"] = sample["image"].astype(np.float32, copy=False) / 255.

        return sample


@manager.TRANSFORMS.add_component
class LowPass(TransformABC):
    """
    Low pass filter image.

    Notice: Apply lowpass to images after normalization.
    """

    def __init__(self, inter_resolution):
        """
        Args:
            inter_resolution: Intermediate resolution.
        """
        self.inter_resolution = inter_resolution

    def __call__(self, sample: Dict) -> Dict:
        """
        """
        image = sample["image"]
        H, W = image.shape[:2]
        image = Image.fromarray(
            (image * 255.).astype(np.uint8, copy=False)).resize(
                (self.inter_resolution, self.inter_resolution)).resize((H, W))
        sample["image"] = np.asarray(image, dtype=np.float32) / 255.

        return sample
