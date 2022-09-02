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

import argparse
import os
import pickle

from paddle3d.datasets.kitti.kitti_depth_det import KittiDepthDataset
from paddle3d.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create infos for kitti dataset.')
    parser.add_argument(
        '--dataset_root',
        default='data/kitti',
        help='Path of the dataset.',
        type=str)
    parser.add_argument(
        '--save_dir',
        default='data/kitti',
        help='Path to save the generated database.',
        type=str)
    return parser.parse_args()


def create_caddn_kitti_infos(dataset, save_path, workers=4):
    train_split, val_split = 'train', 'val'
    train_filename = os.path.join(save_path, 'kitti_infos_train.pkl')
    val_filename = os.path.join(save_path, 'kitti_infos_val.pkl')
    trainval_filename = os.path.join(save_path, 'kitti_infos_trainval.pkl')
    test_filename = os.path.join(save_path, 'kitti_infos_test.pkl')

    logger.info("---------------Start to generate data infos---------------")

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(
        num_workers=workers,
        has_label=True,
        count_inside_pts=True,
        mode='train')
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    logger.info("Kitti info train file is saved to %s" % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(
        num_workers=workers,
        has_label=True,
        count_inside_pts=True,
        mode='train')
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    logger.info("Kitti info val file is saved to %s" % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    logger.info("Kitti info trainval file is saved to %s" % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(
        num_workers=workers,
        has_label=False,
        count_inside_pts=False,
        mode='test')
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    logger.info("Kitti info test file is saved to %s" % test_filename)

    logger.info("---------------Data preparation Done---------------")


def main(args):
    dataset_root = args.dataset_root
    save_dir = args.save_dir
    dataset = KittiDepthDataset(
        dataset_root=dataset_root,
        mode='val',
        point_cloud_range=[2, -30.08, -3.0, 46.8, 30.08, 1.0],
        depth_downsample_factor=4,
        voxel_size=[0.16, 0.16, 0.16],
        class_names=['Car', 'Pedestrian', 'Cyclist'])
    create_caddn_kitti_infos(dataset=dataset, save_path=save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
