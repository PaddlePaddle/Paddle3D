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

import paddle
import tempfile
import warnings
from os import path as osp
import numpy as np
import paddle3d.transforms as T
from paddle3d.apis import manager
import pickle
from paddle3d.geometries.bbox import BBoxes3D, CoordMode
from paddle3d.datasets.base import BaseDataset


def get_box_type():
    """Get the type and mode of box structure.
    """
    box_type_3d = 'LiDAR'
    box_mode_3d = CoordMode.NuScenesLidar
    return box_type_3d, box_mode_3d


@manager.DATASETS.add_component
class Custom3DDataset(BaseDataset):
    """Customized 3D dataset.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 file_client_args=dict(backend='disk')):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type()
        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)  # local path

        if pipeline is not None:
            self.pipeline = T.Compose(pipeline)
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        """
        return pickle.load(ann_file)

    def get_data_info(self, index):
        """Get data info according to the given index.
        """
        info = self.data_infos[index]
        sample_idx = info['sample_idx']
        pts_filename = osp.join(self.data_root,
                                info['lidar_points']['lidar_path'])
        input_dict = dict(
            pts_filename=pts_filename,
            sample_idx=sample_idx,
            file_name=pts_filename)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.
        """
        info = self.data_infos[index]
        gt_bboxes_3d = info['annos']['gt_bboxes_3d']
        gt_names_3d = info['annos']['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        ori_box_type_3d = info['annos']['box_type_3d']
        ori_box_type_3d, _ = get_box_type()
        gt_bboxes_3d = ori_box_type_3d(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def pre_pipeline(self, results):
        """Initialization before data preparation.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def prepare_train_data(self, index):
        """Training data preparation.
        """
        input_dict = self.get_data_info(index)
        # print("input_dict", input_dict)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (example is None
                                     or ~(example['gt_labels_3d'] != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.
        """

        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.
        """
        if classes is None:
            return cls.CLASSES
        if isinstance(classes, str):
            #class_names = mmcv.list_from_file(classes)
            print("check list from file")
            exit()
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        pickle.dump(outputs, out)
        return outputs, tmp_dir

    def __len__(self):
        """Return the length of data infos.

        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    @property
    def metric(self):
        return None

    @property
    def name(self) -> str:
        return None

    @property
    def labels(self):
        return None
