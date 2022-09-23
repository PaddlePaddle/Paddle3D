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

import os
import csv
import numpy as np
from typing import List, Tuple, Union
from paddle3d.geometries import BBoxes2D, BBoxes3D, CoordMode
from typing import List, Tuple
import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import paddle
from paddle.inference import Config, PrecisionType, create_predictor


def parse_args(image_path):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="/home/kerry/project/digital-ai-eye-model/trian/Paddle3D/exported_model/inference.pdmodel",
        help="Model filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="/home/kerry/project/digital-ai-eye-model/trian/Paddle3D/exported_model/inference.pdiparams",
        help="Parameter filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=image_path,
        help="The image path",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Whether use gpu.")
    parser.add_argument("--use_trt", action="store_true", help="Whether use trt.")
    parser.add_argument(
        "--collect_dynamic_shape_info",
        action="store_true",
        help="Whether to collect dynamic shape before using tensorrt.",
    )
    parser.add_argument(
        "--dynamic_shape_file",
        dest="dynamic_shape_file",
        help="The image path",
        type=str,
        default="dynamic_shape_info.txt",
    )
    return parser.parse_args(args=[])


def get_ratio(ori_img_size, output_size, down_ratio=(4, 4)):
    return np.array(
        [
            [
                down_ratio[1] * ori_img_size[1] / output_size[1],
                down_ratio[0] * ori_img_size[0] / output_size[0],
            ]
        ],
        np.float32,
    )


def get_img(img_path):
    img = cv2.imread(img_path)
    origin_shape = img.shape
    img = cv2.resize(img, (1280, 384))

    target_shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    img = np.subtract(img, np.array([0.485, 0.456, 0.406]))
    img = np.true_divide(img, np.array([0.229, 0.224, 0.225]))
    img = np.array(img, np.float32)

    img = img.transpose(2, 0, 1)
    img = img[None, :, :, :]

    return img, origin_shape, target_shape


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
        config.enable_tuned_tensorrt_dynamic_shape(
            args.dynamic_shape_file, allow_build_at_runtime
        )

        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=PrecisionType.Float32,
        )

    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

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


def total_pred_by_conf_to_kitti_records(
    total_pred: np.array, conf: float, class_names=["Car", "Cyclist", "Pedestrian"]
) -> np.array:

    kitti_records_list = []
    for p in total_pred:
        if p[-1] > conf:
            p = list(p)
            p[0] = class_names[int(p[0])]
            p.insert(1, 0.0)
            p.insert(2, 0)
            kitti_records_list.append(p)
    kitti_records = np.array(kitti_records_list)

    return kitti_records


def camera_record_to_object(
    kitti_records: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ """
    if kitti_records.shape[0] == 0:
        bboxes_2d = BBoxes2D(np.zeros([0, 4]))
        bboxes_3d = BBoxes3D(
            np.zeros([0, 7]),
            origin=[0.5, 1, 0.5],
            coordmode=CoordMode.KittiCamera,
            rot_axis=1,
        )
        labels = []
    else:
        centers = kitti_records[:, 11:14]
        dims = kitti_records[:, 8:11]
        yaws = kitti_records[:, 14:15]
        bboxes_3d = BBoxes3D(
            np.concatenate([centers, dims, yaws], axis=1),
            origin=[0.5, 1, 0.5],
            coordmode=CoordMode.KittiCamera,
            rot_axis=1,
        )
        bboxes_2d = BBoxes2D(kitti_records[:, 4:8])
        labels = kitti_records[:, 0]

    return bboxes_2d, bboxes_3d, labels


def encode_label(K, ry, dims, locs):
    """get bbox 3d and 2d by model output
    Args:
        K (np.ndarray): camera intrisic matrix
        ry (np.ndarray): rotation y
        dims (np.ndarray): dimensions
        locs (np.ndarray): locations
    """
    l, h, w = dims[0], dims[1], dims[2]
    x, y, z = locs[0], locs[1], locs[2]

    x_corners = [0, l, l, l, l, 0, 0, 0]
    y_corners = [0, 0, h, h, 0, 0, h, h]
    z_corners = [0, 0, 0, w, w, w, w, 0]

    x_corners += -np.float32(l) / 2
    y_corners += -np.float32(h)
    z_corners += -np.float32(w) / 2

    corners_3d = np.array([x_corners, y_corners, z_corners])
    rot_mat = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )
    corners_3d = np.matmul(rot_mat, corners_3d)
    corners_3d += np.array([x, y, z]).reshape([3, 1])

    loc_center = np.array([x, y - h / 2, z])
    proj_point = np.matmul(K, loc_center)
    proj_point = proj_point[:2] / proj_point[2]

    corners_2d = np.matmul(K, corners_3d)
    corners_2d = corners_2d[:2] / corners_2d[2]
    box2d = np.array(
        [min(corners_2d[0]), min(corners_2d[1]), max(corners_2d[0]), max(corners_2d[1])]
    )

    return proj_point, box2d, corners_3d


def load_calibration_info(calib_path: str) -> Tuple:
    """ """

    with open(os.path.join(calib_path), "r") as csv_file:
        reader = list(csv.reader(csv_file, delimiter=" "))

        # parse camera intrinsics from calibration table
        P0 = [float(i) for i in reader[0][1:]]
        P0 = np.array(P0, dtype=np.float32).reshape(3, 4)

        P1 = [float(i) for i in reader[1][1:]]
        P1 = np.array(P1, dtype=np.float32).reshape(3, 4)

        P2 = [float(i) for i in reader[2][1:]]
        P2 = np.array(P2, dtype=np.float32).reshape(3, 4)

        P3 = [float(i) for i in reader[3][1:]]
        P3 = np.array(P3, dtype=np.float32).reshape(3, 4)

        # parse correction matrix for camera 0.
        R0_rect = [float(i) for i in reader[4][1:]]
        R0_rect = np.array(R0_rect, dtype=np.float32).reshape(3, 3)

        # parse matrix from velodyne to camera
        V2C = [float(i) for i in reader[5][1:]]
        V2C = np.array(V2C, dtype=np.float32).reshape(3, 4)

        if len(reader) == 6:
            # parse matrix from imu to velodyne
            I2V = [float(i) for i in reader[6][1:]]
            I2V = np.array(I2V, dtype=np.float32).reshape(3, 4)
        else:
            I2V = np.array([0, 4], dtype=np.float32)

    return P0, P1, P2, P3, R0_rect, V2C, I2V


def convert_to_calib_path(img_path: str) -> str:
    """ """

    img_name = os.path.basename(img_path)
    calib_name = img_name[: img_name.rindex(".")] + ".txt"
    img_dirname = os.path.dirname(img_path)
    return os.path.join(img_dirname[: img_dirname.rindex("/")], "calib", calib_name)


if __name__ == "__main__":
    img_path = "/home/kerry/project/digital-ai-eye-model/trian/Paddle3D/datasets/KITTI/training/image_2/000006.png"
    args = parse_args(img_path)
    pred = init_predictor(args)
    # Listed below are camera intrinsic parameter of the kitti dataset
    # If the model is trained on other datasets, please replace the relevant data
    K = np.array(
        [
            [
                [721.53771973, 0.0, 609.55932617],
                [0.0, 721.53771973, 172.85400391],
                [0, 0, 1],
            ]
        ],
        np.float32,
    )

    img, ori_img_size, output_size = get_img(args.image)
    ratio = get_ratio(ori_img_size, output_size)

    results = run(pred, [img, K, ratio])

    total_pred = results[0]

    # print(total_pred)
    print((total_pred))
    np.save("total_pred.npy", total_pred)
