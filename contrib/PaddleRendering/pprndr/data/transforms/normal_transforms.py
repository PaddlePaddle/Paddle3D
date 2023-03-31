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

import numpy as np
from PIL import Image

from pprndr.apis import manager
from pprndr.data.transforms.base import TransformABC
from pprndr.utils.logger import logger

__all__ = ["LoadNormal"]


@manager.TRANSFORMS.add_component
class LoadNormal(TransformABC):
    """
    Load normal from file path, default backend is pillow.
    """

    @staticmethod
    def pillow_reader(normal_path):
        return np.asarray(
            Image.open(normal_path), dtype=np.uint8).astype(
                np.float32, copy=False)

    def __call__(self, sample: Dict) -> Dict:
        """
        """
        sample["normal"] = LoadNormal.pillow_reader(sample["normal"])

        return sample
