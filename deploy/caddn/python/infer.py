# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import inference
from paddle.static import InputSpec
from skimage import io

from paddle3d.apis.config import Config
from paddle3d.ops.iou3d_nms_cuda import nms_gpu


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
        '--img_path', type=str, help='The image path.', required=True)
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


def load_predictor(model_file,
                   params_file,
                   gpu_id=0,
                   use_trt=False,
                   trt_precision=0,
                   trt_use_static=False,
                   trt_static_dir=None,
                   collect_shape_info=False,
                   dynamic_shape_file=None):
    """load_predictor
    initialize the inference engine
    """
    config = inference.Config(model_file, params_file)
    config.enable_use_gpu(1000, gpu_id)

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    if use_trt:
        precision_mode = paddle.inference.PrecisionType.Float32
        if trt_precision == 1:
            precision_mode = paddle.inference.PrecisionType.Half
        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=1,
            min_subgraph_size=30,
            precision_mode=precision_mode,
            use_static=trt_use_static,
            use_calib_mode=False)
        if collect_shape_info:
            config.collect_shape_range_info(dynamic_shape_file)
        else:
            config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
        if trt_use_static:
            config.set_optim_cache_dir(trt_static_dir)
    predictor = inference.create_predictor(config)

    return predictor


def get_image(img_path):
    """
    Loads image for a sample
    Args:
        idx [int]: Index of the image sample
    Returns:
        image [np.ndarray(H, W, 3)]: RGB Image
    """
    assert os.path.exists(img_path)
    image = io.imread(img_path)
    image = image[:, :, :3]  # Remove alpha channel
    image = image.astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    image = image.transpose([0, 3, 1, 2])
    return image


def run(predictor, img):
    input_names = predictor.get_input_names()

    input_tensor1 = predictor.get_input_handle(input_names[0])
    input_tensor2 = predictor.get_input_handle(input_names[1])
    input_tensor3 = predictor.get_input_handle(input_names[2])

    data = {}
    data["images"] = img
    data["trans_lidar_to_cam"] = np.asarray(
        [[[0.0048523, -0.9999298, -0.01081266, -0.00711321],
          [-0.00302069, 0.01079808, -0.99993706, -0.06176636],
          [0.99998367, 0.00488465, -0.00296808, -0.26739058],
          [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]],
        dtype='float32')
    data["trans_cam_to_img"] = np.asarray(
        [[[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01],
          [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01],
          [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]]],
        dtype='float32')
    input_tensor1.copy_from_cpu(data[input_names[0]])
    input_tensor2.copy_from_cpu(data[input_names[1]])
    input_tensor3.copy_from_cpu(data[input_names[2]])
    predictor.run()
    outs = []
    output_names = predictor.get_output_names()
    for name in output_names:
        out = predictor.get_output_handle(name)
        out = out.copy_to_cpu()
        out = paddle.to_tensor(out)
        outs.append(out)

    result = {}
    result['pred_boxes'] = outs[0]
    result['pred_labels'] = outs[1]
    result['pred_scores'] = outs[2]

    return result


def main(args):
    predictor = load_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.collect_shape_info, args.dynamic_shape_file)
    image = get_image(args.img_path)
    result = run(predictor, image)
    print(result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
