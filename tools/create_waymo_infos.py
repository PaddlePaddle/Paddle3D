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
import multiprocessing
import os
import pickle
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm

from paddle3d.datasets.waymo import WaymoPCDataset
from paddle3d.geometries import BBoxes3D
from paddle3d.geometries.bbox import get_mask_of_points_in_bboxes3d
from paddle3d.utils.logger import logger


def create_waymo_gt_database(dataset_root,
                             class_names,
                             save_path=None,
                             sampled_interval=1,
                             use_point_dim=5):
    if save_path is None:
        save_path = dataset_root

    save_path = os.path.join(save_path, "waymo_train_gt_database")

    dataset = WaymoPCDataset(
        dataset_root=dataset_root,
        sampled_interval=sampled_interval,
        mode="train",
        class_names=class_names)
    database = defaultdict(list)

    for data_idx in tqdm(range(0, len(dataset), sampled_interval)):
        sample = dataset[data_idx]
        points = sample.data
        bboxes_3d = sample.bboxes_3d
        labels = sample.labels  # starts from 0
        difficulties = sample.difficulties
        box_names = np.array(class_names)[labels]

        # sampling 1/4 of "Vehicle" class
        if data_idx % 4 != 0 and len(box_names) > 0:
            mask = (box_names == "Vehicle")
            box_names = box_names[~mask]
            difficulties = difficulties[~mask]
            bboxes_3d = BBoxes3D(
                data=bboxes_3d[~mask],
                coordmode=bboxes_3d.coordmode,
                origin=bboxes_3d.origin)

        # sampling 1/2 of "Pedestrian" class
        if data_idx % 2 != 0 and len(box_names) > 0:
            mask = (box_names == "Pedestrian")
            box_names = box_names[~mask]
            difficulties = difficulties[~mask]
            bboxes_3d = BBoxes3D(
                data=bboxes_3d[~mask],
                coordmode=bboxes_3d.coordmode,
                origin=bboxes_3d.origin)

        num_bboxes = len(bboxes_3d)
        if num_bboxes == 0:
            continue

        # TODO(liuxiao): get_mask could be accelerate
        masks = get_mask_of_points_in_bboxes3d(points, bboxes_3d)
        for box_idx in range(num_bboxes):
            box_name = box_names[box_idx]
            if box_name not in class_names:
                continue
            mask = masks[:, box_idx]
            selected_points = points[mask]
            selected_points[:, :3] -= bboxes_3d[box_idx, :3]

            if not os.path.exists(os.path.join(save_path, box_name)):
                os.makedirs(os.path.join(save_path, box_name))
            lidar_file = os.path.join(
                os.path.join(save_path, box_name), "{}_{}_{}.bin".format(
                    data_idx, box_name, box_idx))

            with open(lidar_file, "w") as f:
                selected_points.tofile(f)

            anno_info = {
                "lidar_file":
                os.path.join("waymo_train_gt_database",
                             box_name, "{}_{}_{}.bin".format(
                                 data_idx, box_name, box_idx)),
                "cls_name":
                box_name,
                "bbox_3d":
                bboxes_3d[box_idx, :],
                "box_idx":
                box_idx,
                "data_idx":
                data_idx,
                "num_points_in_box":
                selected_points.shape[0],
                "lidar_dim":
                use_point_dim,
                "difficulty":
                difficulties[box_idx]
            }
            database[box_name].append(anno_info)

    for k, v in database.items():
        logger.info("Database %s: %d" % (k, len(v)))

    db_anno_file = os.path.join(save_path, "waymo_train_gt_database_infos.pkl")
    with open(db_anno_file, 'wb') as f:
        pickle.dump(database, f)


def get_infos(raw_data_path,
              save_path,
              sample_sequence_list,
              num_workers=multiprocessing.cpu_count(),
              sampled_interval=1):

    from functools import partial

    from paddle3d.datasets.waymo import waymo_utils
    logger.info(
        "---------------The waymo sample interval is %d, total sequecnes is %d-----------------"
        % (sampled_interval, len(sample_sequence_list)))

    process_single_sequence = partial(
        waymo_utils.process_single_sequence,
        save_path=save_path,
        sampled_interval=sampled_interval)

    sample_sequence_file_list = [
        os.path.join(raw_data_path, sequence_file)
        for sequence_file in sample_sequence_list
    ]

    p = multiprocessing.Pool(num_workers)
    sequence_infos = list(
        tqdm(
            p.map(process_single_sequence, sample_sequence_file_list),
            total=len(sample_sequence_file_list)))
    p.close()
    p.join()

    all_sequences_infos = [item for infos in sequence_infos for item in infos]
    return all_sequences_infos


def create_waymo_infos(dataset_root,
                       class_names,
                       save_path,
                       raw_data_tag,
                       processed_data_tag,
                       num_workers=min(16, multiprocessing.cpu_count())):

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger.info("---------------Start to generate data infos---------------")

    dataset = WaymoPCDataset(
        dataset_root=dataset_root,
        sampled_interval=1,
        mode="train",
        class_names=class_names,
        processed_data_tag=processed_data_tag)
    waymo_infos_train = get_infos(
        raw_data_path=os.path.join(dataset_root, raw_data_tag),
        save_path=os.path.join(save_path, processed_data_tag),
        sample_sequence_list=dataset.sample_sequence_list,
        num_workers=num_workers,
        sampled_interval=1  # save all infos
    )
    logger.info("----------------Waymo train info is saved-----------------")

    dataset = WaymoPCDataset(
        dataset_root=dataset_root,
        sampled_interval=1,
        mode="val",
        class_names=class_names,
        processed_data_tag=processed_data_tag)
    waymo_infos_val = get_infos(
        raw_data_path=os.path.join(dataset_root, raw_data_tag),
        save_path=os.path.join(save_path, processed_data_tag),
        sample_sequence_list=dataset.sample_sequence_list,
        num_workers=num_workers,
        sampled_interval=1  # save all infos
    )
    logger.info("----------------Waymo val info is saved-----------------")

    logger.info("-------------------Create gt database-------------------")

    create_waymo_gt_database(
        dataset_root=dataset_root,
        class_names=class_names,
        save_path=save_path,
        sampled_interval=1,  # sampling all gt
        use_point_dim=5)
    logger.info("-------------------Create gt database done-------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create infos and gt database")
    parser.add_argument(
        "--processed_data_tag",
        type=str,
        default="waymo_processed_data_v1_3_2",
        help="")
    args = parser.parse_args()

    create_waymo_infos(
        dataset_root="./datasets/waymo",
        class_names=["Vehicle", "Pedestrian", "Cyclist"],
        save_path="./datasets/waymo",
        raw_data_tag="raw_data",
        processed_data_tag=args.processed_data_tag)
