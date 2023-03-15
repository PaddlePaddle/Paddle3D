import argparse

import cv2
import numpy as np

from paddle.inference import Config, PrecisionType, create_predictor
from paddle3d.datasets.kitti.kitti_utils import camera_record_to_object
from utils import get_img, get_ratio, total_pred_by_conf_to_kitti_records, make_imgpts_list, draw_mono_3d


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
        '--image', dest='image', help='The image path', type=str, required=True)
    parser.add_argument(
        "--use_gpu", action='store_true', help="Whether use gpu.")
    parser.add_argument(
        "--use_trt", action='store_true', help="Whether use trt.")
    parser.add_argument(
        "--collect_dynamic_shape_info",
        action='store_true',
        help="Whether to collect dynamic shape before using tensorrt.")
    parser.add_argument(
        "--dynamic_shape_file",
        dest='dynamic_shape_file',
        help='The image path',
        type=str,
        default="dynamic_shape_info.txt")
    return parser.parse_args()


def init_predictor(args):
    config = Config(args.model_file, args.params_file)
    config.enable_memory_optim()
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
    else:
        # If not specific mkldnn, you can set the blas thread.
        # The thread num should not be greater than the number of cores in the CPU.
        config.set_cpu_math_library_num_threads(4)
        config.enable_mkldnn()

    if args.collect_dynamic_shape_info:
        config.collect_shape_range_info(args.dynamic_shape_file)
    elif args.use_trt:
        allow_build_at_runtime = True
        config.enable_tuned_tensorrt_dynamic_shape(args.dynamic_shape_file,
                                                   allow_build_at_runtime)

        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=PrecisionType.Float32)

    predictor = create_predictor(config)
    return predictor


def run(predictor, image, K, down_ratio):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        if name == "images":
            input_tensor.reshape(image.shape)
            input_tensor.copy_from_cpu(image.copy())
        elif name == "trans_cam_to_img":
            input_tensor.reshape(K.shape)
            input_tensor.copy_from_cpu(K.copy())
        elif name == "down_ratios":
            input_tensor.reshape(down_ratio.shape)
            input_tensor.copy_from_cpu(down_ratio.copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


if __name__ == '__main__':
    args = parse_args()
    pred = init_predictor(args)
    # Listed below are camera intrinsic parameter of the kitti dataset
    # If the model is trained on other datasets, please replace the relevant data
    K = np.array([[[721.53771973, 0., 609.55932617],
                   [0., 721.53771973, 172.85400391], [0, 0, 1]]], np.float32)

    img, ori_img_size, output_size = get_img(args.image)
    ratio = get_ratio(ori_img_size, output_size)

    results = run(pred, img, K, ratio)

    total_pred = results[0]
    print(total_pred)
    # convert pred to bboxes_2d, bboxes_3d
    kitti_records = total_pred_by_conf_to_kitti_records(total_pred, conf=0.5)
    bboxes_2d, bboxes_3d, labels = camera_record_to_object(kitti_records)
    # read origin image
    img_origin = cv2.imread(args.image)
    # to 8 points on image
    imgpts_list = make_imgpts_list(bboxes_3d, K[0])
    # draw smoke result to photo
    draw_mono_3d(img_origin, imgpts_list)
