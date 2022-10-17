import argparse
import os

import numpy as np
import paddle
from paddle.inference import Config, create_predictor

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
        "--use_trt",
        type=int,
        default=0,
        help="Whether to use tensorrt to accelerate when using gpu.")
    parser.add_argument(
        "--trt_precision",
        type=int,
        default=0,
        help="Precision type of tensorrt, 0: kFloat32, 1: kHalf.")
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
        choice = []
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
    # stack batch_idx into points
    batch_idx = np.zeros((num_points, 1), dtype=np.float32)
    points = np.concatenate([batch_idx, points], axis=-1)
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
        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=1,
            min_subgraph_size=10,
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


def run(predictor, points):
    # copy points data into input_tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        if name == "data":
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(points.shape)
            input_tensor.copy_from_cpu(points.copy())

    # do the inference
    predictor.run()

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
    np.random.seed(1024)
    predictor = init_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir,
                               args.collect_shape_info, args.dynamic_shape_file)
    num_points = 16384
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    points = preprocess(args.lidar_file, num_points, point_cloud_range)
    box3d_lidar, label_preds, scores = run(predictor, points)
    print({'boxes': box3d_lidar, 'labels': label_preds, 'scores': scores})


if __name__ == '__main__':
    args = parse_args()
    main(args)
