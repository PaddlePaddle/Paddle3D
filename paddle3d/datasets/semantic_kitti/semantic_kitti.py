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

__all__ = ["SemanticKITTIDataset"]

import os
from glob import glob
from typing import List, Tuple, Union

import numpy as np

from paddle3d import transforms as T
from paddle3d.datasets import BaseDataset
from paddle3d.transforms import TransformABC


class SemanticKITTIDataset(BaseDataset):
    """
    SemanticKITTI dataset.

    Class attributes (`LABELS`, `LEARNING_MAP`, `LEARNING_MAP_INV`, `CONTENT`,
    `LEARNING_IGNORE`, `SEQUENCE_SPLITS`) are from SemanticKITTI dataset official
    configuration. Please refer to:
    <https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti-all.yaml>.

    Args:
        dataset_root (str): Path to the root directory of SemanticKITTI dataset.
        mode (str, optional): The mode of dataset. Default is 'train'.
        sequences (list or tuple, optional): The data sequences of dataset.
            If None, use default sequence splits according to `mode`. Default is None.
        transforms (TransformABC or list[TransformABC], optional): The transforms of dataset. Default is None.
    """

    LABELS = {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle"
    }

    LEARNING_MAP = {
        0: 0,  # "unlabeled"
        1: 0,  # "outlier" mapped to "unlabeled" ------------------------mapped
        10: 1,  # "car"
        11: 2,  # "bicycle"
        13: 5,  # "bus" mapped to "other-vehicle" ------------------------mapped
        15: 3,  # "motorcycle"
        16: 5,  # "on-rails" mapped to "other-vehicle" -------------------mapped
        18: 4,  # "truck"
        20: 5,  # "other-vehicle"
        30: 6,  # "person"
        31: 7,  # "bicyclist"
        32: 8,  # "motorcyclist"
        40: 9,  # "road"
        44: 10,  # "parking"
        48: 11,  # "sidewalk"
        49: 12,  # "other-ground"
        50: 13,  # "building"
        51: 14,  # "fence"
        52: 0,  # "other-structure" mapped to "unlabeled" ----------------mapped
        60: 9,  # "lane-marking" to "road" -------------------------------mapped
        70: 15,  # "vegetation"
        71: 16,  # "trunk"
        72: 17,  # "terrain"
        80: 18,  # "pole"
        81: 19,  # "traffic-sign"
        99: 0,  # "other-object" to "unlabeled" --------------------------mapped
        252:
        1,  # "moving-car" to "car" ----------------------------------mapped
        253:
        7,  # "moving-bicyclist" to "bicyclist" ----------------------mapped
        254:
        6,  # "moving-person" to "person" ----------------------------mapped
        255:
        8,  # "moving-motorcyclist" to "motorcyclist" ----------------mapped
        256:
        5,  # "moving-on-rails" mapped to "other-vehicle" ------------mapped
        257:
        5,  # "moving-bus" mapped to "other-vehicle" -----------------mapped
        258:
        4,  # "moving-truck" to "truck" ------------------------------mapped
        259:
        5,  # "moving-other"-vehicle to "other-vehicle" --------------mapped
    }

    LEARNING_MAP_INV = {  # inverse of previous map
        0: 0,  # "unlabeled", and others ignored
        1: 10,  # "car"
        2: 11,  # "bicycle"
        3: 15,  # "motorcycle"
        4: 18,  # "truck"
        5: 20,  # "other-vehicle"
        6: 30,  # "person"
        7: 31,  # "bicyclist"
        8: 32,  # "motorcyclist"
        9: 40,  # "road"
        10: 44,  # "parking"
        11: 48,  # "sidewalk"
        12: 49,  # "other-ground"
        13: 50,  # "building"
        14: 51,  # "fence"
        15: 70,  # "vegetation"
        16: 71,  # "trunk"
        17: 72,  # "terrain"
        18: 80,  # "pole"
        19: 81,  # "traffic-sign"
    }

    CONTENT = {  # as a ratio with the total number of points
        0: 0.018889854628292943,
        1: 0.0002937197336781505,
        10: 0.040818519255974316,
        11: 0.00016609538710764618,
        13: 2.7879693665067774e-05,
        15: 0.00039838616015114444,
        16: 0.0,
        18: 0.0020633612104619787,
        20: 0.0016218197275284021,
        30: 0.00017698551338515307,
        31: 1.1065903904919655e-08,
        32: 5.532951952459828e-09,
        40: 0.1987493871255525,
        44: 0.014717169549888214,
        48: 0.14392298360372,
        49: 0.0039048553037472045,
        50: 0.1326861944777486,
        51: 0.0723592229456223,
        52: 0.002395131480328884,
        60: 4.7084144280367186e-05,
        70: 0.26681502148037506,
        71: 0.006035012012626033,
        72: 0.07814222006271769,
        80: 0.002855498193863172,
        81: 0.0006155958086189918,
        99: 0.009923127583046915,
        252: 0.001789309418528068,
        253: 0.00012709999297008662,
        254: 0.00016059776092534436,
        255: 3.745553104802113e-05,
        256: 0.0,
        257: 0.00011351574470342043,
        258: 0.00010157861367183268,
        259: 4.3840131989471124e-05,
    }

    LEARNING_IGNORE = {
        0: True,  # "unlabeled", and others ignored
        1: False,  # "car"
        2: False,  # "bicycle"
        3: False,  # "motorcycle"
        4: False,  # "truck"
        5: False,  # "other-vehicle"
        6: False,  # "person"
        7: False,  # "bicyclist"
        8: False,  # "motorcyclist"
        9: False,  # "road"
        10: False,  # "parking"
        11: False,  # "sidewalk"
        12: False,  # "other-ground"
        13: False,  # "building"
        14: False,  # "fence"
        15: False,  # "vegetation"
        16: False,  # "trunk"
        17: False,  # "terrain"
        18: False,  # "pole"
        19: False,  # "traffic-sign"
    }

    SEQUENCE_SPLITS = {
        'train': (0, 1, 2, 3, 4, 5, 6, 7, 9, 10),
        'val': (8, ),
        'test': (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    }

    def __init__(self,
                 dataset_root: str,
                 mode: str = "train",
                 sequences: Union[List[int], Tuple[int], None] = None,
                 transforms: Union[TransformABC, List[TransformABC]] = None):
        super().__init__()
        self.mode = mode

        if isinstance(transforms, list):
            transforms = T.Compose(transforms)

        self.transforms = transforms

        if self.mode not in ['train', 'val', 'trainval', 'test']:
            raise ValueError(
                "mode should be 'train', 'val', 'trainval' or 'test', but got {}."
                .format(self.mode))

        if sequences is not None:
            self.sequences = sequences
        else:
            self.sequences = self.SEQUENCE_SPLITS[self.mode]

        # get file list
        self.data = []
        for seq in self.sequences:
            seq_dir = os.path.join(dataset_root, 'sequences', '{0:02d}'.format(
                int(seq)))
            scans = sorted(glob(os.path.join(seq_dir, 'velodyne', '*.bin')))
            self.data.extend(scans)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def build_remap_lut():
        """
        Make lookup table for mapping
        """

        maxkey = max(SemanticKITTIDataset.LEARNING_MAP.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(SemanticKITTIDataset.LEARNING_MAP.keys())] = list(
            SemanticKITTIDataset.LEARNING_MAP.values())

        return remap_lut

    @property
    def name(self) -> str:
        return "SemanticKITTI"

    @property
    def labels(self) -> List[str]:
        num_classes = len(self.LEARNING_MAP_INV)
        class_names = [
            self.LABELS[self.LEARNING_MAP_INV[i]] for i in range(num_classes)
        ]
        return class_names
