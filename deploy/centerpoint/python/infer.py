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

import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor

from paddle3d.ops.centerpoint_postprocess import centerpoint_postprocess
from paddle3d.ops.voxelize import hard_voxelize


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
    parser.add_argument(
        "--num_point_dim",
        type=int,
        default=4,
        help="Dimension of a point in the lidar file.")
    parser.add_argument(
        "--use_timelag",
        type=int,
        default=0,
        help="Whether to insert timelag to each point.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU card id.")
    parser.add_argument(
        "--use_trt",
        type=int,
        default=0,
        help="Whether to use tensorrt to accelerate when using gpu.")
    parser.add_argument(
        "--trt_precision",
        type=int,
        default=0,
        help="Precision type of tensorrt, 0: kFloat32, 1: kHalf., 2: Int8")
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
    parser.add_argument(
        "--collect_shape_info",
        type=int,
        default=0,
        help="Whether to collect dynamic shape before using tensorrt.")
    parser.add_argument(
        "--dynamic_shape_file",
        type=str,
        default="",
        help="Path of a dynamic shape file for tensorrt.")

    return parser.parse_args()


def read_point(file_path, num_point_dim):
    points = np.fromfile(file_path, np.float32).reshape(-1, num_point_dim)
    points = points[:, :4]
    return points


def insert_time_to_points(points):
    time_lag = np.zeros((points.shape[0], 1), dtype=points.dtype)
    points = np.hstack([points, time_lag])
    return points


def preprocess(file_path, num_point_dim, use_timelag):
    points = read_point(file_path, num_point_dim)
    if use_timelag:
        points = insert_time_to_points(points)
    return points


def init_predictor(model_file,
                   params_file,
                   gpu_id=0,
                   use_trt=False,
                   trt_precision=0,
                   trt_use_static=False,
                   trt_static_dir=None,
                   collect_shape_info=False,
                   dynamic_shape_file=None):
    config = Config(model_file, params_file)
    config.enable_memory_optim()
    config.enable_use_gpu(1000, gpu_id)
    if use_trt:
        precision_mode = paddle.inference.PrecisionType.Float32
        if trt_precision == 1:
            precision_mode = paddle.inference.PrecisionType.Half
        elif trt_precision == 2:
            precision_mode = paddle.inference.PrecisionType.Int8
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=precision_mode,
            use_static=trt_use_static,
            use_calib_mode=False)
        if collect_shape_info:
            config.collect_shape_range_info(dynamic_shape_file)
        else:
            config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
        if trt_use_static:
            config.set_optim_cache_dir(trt_static_dir)

    predictor = create_predictor(config)
    return predictor


def parse_result(box3d_lidar, label_preds, scores):
    num_bbox3d, bbox3d_dims = box3d_lidar.shape
    for box_idx in range(num_bbox3d):
        # filter fake results: score = -1
        if scores[box_idx] < 0:
            continue
        if bbox3d_dims == 9:
            print(
                "Score: {} Label: {} Box(x_c, y_c, z_c, w, l, h, vec_x, vec_y, -rot): {} {} {} {} {} {} {} {} {}"
                .format(scores[box_idx], label_preds[box_idx],
                        box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1],
                        box3d_lidar[box_idx, 2], box3d_lidar[box_idx, 3],
                        box3d_lidar[box_idx, 4], box3d_lidar[box_idx, 5],
                        box3d_lidar[box_idx, 6], box3d_lidar[box_idx, 7],
                        box3d_lidar[box_idx, 8]))
        elif bbox3d_dims == 7:
            print(
                "Score: {} Label: {} Box(x_c, y_c, z_c, w, l, h, -rot): {} {} {} {} {} {} {}"
                .format(scores[box_idx], label_preds[box_idx],
                        box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1],
                        box3d_lidar[box_idx, 2], box3d_lidar[box_idx, 3],
                        box3d_lidar[box_idx, 4], box3d_lidar[box_idx, 5],
                        box3d_lidar[box_idx, 6]))


def run(predictor, points):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        if name == "data":
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(points.shape)
            input_tensor.copy_from_cpu(points.copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        if i == 0:
            box3d_lidar = output_tensor.copy_to_cpu()
        elif i == 1:
            label_preds = output_tensor.copy_to_cpu()
        elif i == 2:
            scores = output_tensor.copy_to_cpu()
    return box3d_lidar, label_preds, scores


def main(args):
    predictor = init_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir,
                               args.collect_shape_info, args.dynamic_shape_file)
    points = preprocess(args.lidar_file, args.num_point_dim, args.use_timelag)
    box3d_lidar, label_preds, scores = run(predictor, points)
    parse_result(box3d_lidar, label_preds, scores)


if __name__ == '__main__':
    args = parse_args()

    main(args)
