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

import numpy as np
import paddle
from paddle.inference import Config, PrecisionType, create_predictor

from paddle3d.ops.iou3d_nms_cuda import nms_gpu
from paddle3d.ops.pointnet2_ops import (ball_query, farthest_point_sample,
                                        gather_operation, group_operation)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        help="Model filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        "--params_file",
        type=str,
        help=
        "Parameter filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        '--lidar_file', type=str, help='The lidar path.', required=True)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU card id.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="",
        help="Run_mode which can be: trt_fp32, trt_fp16, trt_int8 and gpu_fp16."
    )
    parser.add_argument(
        "--trt_use_static",
        type=int,
        default=0,
        help="Whether to load the tensorrt graph optimization from a disk path."
    )
    parser.add_argument(
        "--trt_static_dir",
        type=str,
        help="Path of a tensorrt graph optimization directory.")

    return parser.parse_args()


def read_point(lidar_file):
    points = np.fromfile(lidar_file, np.float32).reshape(-1, 4)
    return points


def filter_points_outside_range(points, point_cloud_range):
    limit_range = np.asarray(point_cloud_range, dtype=np.float32)
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    points = points[mask]
    return points


def sample_point(points, num_points):
    if num_points < len(points):
        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        if num_points > len(far_idxs_choice):
            near_idxs_choice = np.random.choice(
                near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            choice = np.random.choice(choice, num_points, replace=False)
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, len(points), dtype=np.int32)
        if num_points > len(points):
            extra_choice = np.random.choice(choice, num_points - len(points))
            choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)
    points = points[choice]
    return points


def preprocess(lidar_file, num_points, point_cloud_range):
    points = read_point(lidar_file)
    points = filter_points_outside_range(points, point_cloud_range)
    points = sample_point(points, num_points)
    return points


def init_predictor(model_file,
                   params_file,
                   gpu_id=0,
                   run_mode=None,
                   trt_use_static=False,
                   trt_static_dir=None):
    config = Config(model_file, params_file)
    config.enable_memory_optim()
    config.enable_use_gpu(1000, gpu_id)
    if args.run_mode == "gpu_fp16":
        config.exp_enable_use_gpu_fp16()
    elif args.run_mode == "trt_fp32":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=15,
            precision_mode=PrecisionType.Float32,
            use_static=trt_use_static,
            use_calib_mode=False)
    elif args.run_mode == "trt_fp16":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=15,
            precision_mode=PrecisionType.Half,
            use_static=trt_use_static,
            use_calib_mode=False)
    elif args.run_mode == "trt_int8":
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=15,
            precision_mode=PrecisionType.Int8,
            use_static=trt_use_static,
            use_calib_mode=True)
    if trt_use_static:
        config.set_optim_cache_dir(trt_static_dir)

    predictor = create_predictor(config)
    return predictor


def run(predictor, points):
    # copy points data into input_tensor
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.reshape(points.shape)
    input_tensor.copy_from_cpu(points.copy())

    # do the inference
    predictor.run()

    # get out data from output tensor
    output_names = predictor.get_output_names()
    return [
        predictor.get_output_handle(name).copy_to_cpu() for name in output_names
    ]


def main(args):
    np.random.seed(1024)
    predictor = init_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.run_mode)
    num_points = 16384
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]

    points = preprocess(args.lidar_file, num_points, point_cloud_range)
    box3d_lidar, label_preds, scores = run(predictor, points)
    print({'boxes': box3d_lidar, 'labels': label_preds, 'scores': scores})


if __name__ == '__main__':
    args = parse_args()
    main(args)
