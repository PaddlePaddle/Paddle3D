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

from .base import BaseDataset
from .kitti import KittiDepthDataset, KittiMonoDataset, KittiPCDataset
from .modelnet40 import ModelNet40
from .nuscenes import NuscenesMVDataset, NuscenesPCDataset, NuscenesMVSegDataset
from .waymo import WaymoPCDataset
from .apollo import ApolloOffsetDataset, ApolloOffsetValDataset
from paddle3d.apis import manager

# for PaddleX
NuscenesPCDetDataset = NuscenesPCDataset
KittiDepthMonoDetDataset = KittiDepthDataset
NuscenesMVDetDataset = NuscenesMVDataset
manager.DATASETS._components_dict['NuscenesPCDetDataset'] = NuscenesPCDetDataset
manager.DATASETS._components_dict[
    'KittiDepthMonoDetDataset'] = KittiDepthMonoDetDataset
manager.DATASETS._components_dict['NuscenesMVDetDataset'] = NuscenesMVDetDataset
