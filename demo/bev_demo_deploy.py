import argparse
import numpy as np
import paddle
from paddle.inference import Config, create_predictor
from paddle3d.ops.iou3d_nms_cuda import nms_gpu
from utils import preprocess, Calibration, show_bev_with_boxes


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
        '--calib_file', type=str, help='The lidar path.', required=True)
    parser.add_argument(
        "--num_point_dim",
        type=int,
        default=4,
        help="Dimension of a point in the lidar file.")
    parser.add_argument(
        "--point_cloud_range",
        dest='point_cloud_range',
        nargs='+',
        help="Range of point cloud for voxelize operation.",
        type=float,
        default=None)
    parser.add_argument(
        "--voxel_size",
        dest='voxel_size',
        nargs='+',
        help="Size of voxels for voxelize operation.",
        type=float,
        default=None)
    parser.add_argument(
        "--max_points_in_voxel",
        type=int,
        default=100,
        help="Maximum number of points in a voxel.")
    parser.add_argument(
        "--max_voxel_num",
        type=int,
        default=12000,
        help="Maximum number of voxels.")
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
            workspace_size=1 << 30,
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


def run(predictor, voxels, coords, num_points_per_voxel):
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        if name == "voxels":
            input_tensor.reshape(voxels.shape)
            input_tensor.copy_from_cpu(voxels.copy())
        elif name == "coords":
            input_tensor.reshape(coords.shape)
            input_tensor.copy_from_cpu(coords.copy())
        elif name == "num_points_per_voxel":
            input_tensor.reshape(num_points_per_voxel.shape)
            input_tensor.copy_from_cpu(num_points_per_voxel.copy())

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


if __name__ == '__main__':
    args = parse_args()

    predictor = init_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir,
                               args.collect_shape_info, args.dynamic_shape_file)
    voxels, coords, num_points_per_voxel = preprocess(
        args.lidar_file, args.num_point_dim, args.point_cloud_range,
        args.voxel_size, args.max_points_in_voxel, args.max_voxel_num)
    box3d_lidar, label_preds, scores = run(predictor, voxels, coords,
                                           num_points_per_voxel)

    scan = np.fromfile(args.lidar_file, dtype=np.float32)
    pc_velo = scan.reshape((-1, 4))

    # Obtain calibration information about Kitti
    calib = Calibration(args.calib_file)

    # Plot box in lidar cloud
    show_bev_with_boxes(pc_velo, box3d_lidar, scores, calib)
