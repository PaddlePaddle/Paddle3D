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

from paddle3d.datasets.generate_gt_database import (
    generate_kitti_gt_database, generate_nuscenes_gt_database)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a ground truth database for a dataset.')
    parser.add_argument(
        '--dataset_name',
        dest='dataset_name',
        help='Name of the dataset: nuscenes, or kitti.',
        type=str)
    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='Path of the dataset.',
        type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='Path to save the generated database.',
        type=str)
    return parser.parse_args()


def main(args):
    if args.dataset_name.lower() == 'nuscenes':
        generate_nuscenes_gt_database(args.dataset_root, save_dir=args.save_dir)
    elif args.dataset_name.lower() == 'kitti':
        generate_kitti_gt_database(args.dataset_root, save_dir=args.save_dir)
    else:
        raise ValueError(
            f"Database generation is not supported for the {args.dataset_name} dataset."
        )


if __name__ == '__main__':
    args = parse_args()
    main(args)
