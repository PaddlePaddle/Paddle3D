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

import cv2
import numpy as np
from infer import get_img, get_ratio, init_predictor, parse_args, run

from paddle3d.datasets.kitti.kitti_utils import camera_record_to_object
from paddle3d.transforms.target_generator import encode_label


def total_pred_by_conf_to_kitti_records(
        total_pred, conf, class_names=["Car", "Cyclist", "Pedestrian"]):
    """convert total_pred to kitti_records"""
    kitti_records_list = []
    for p in total_pred:
        if p[-1] > conf:
            p = list(p)
            p[0] = class_names[int(p[0])]
            # default, to kitti_records formate
            p.insert(1, 0.0)
            p.insert(2, 0)
            kitti_records_list.append(p)
    kitti_records = np.array(kitti_records_list)

    return kitti_records


def make_imgpts_list(bboxes_3d, K):
    """to 8 points on image"""
    # external parameters do not transform
    rvec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tvec = np.array([[0.0], [0.0], [0.0]])

    imgpts_list = []
    for box3d in bboxes_3d:

        locs = np.array(box3d[0:3])
        rot_y = np.array(box3d[6])

        height, width, length = box3d[3:6]
        _, box2d, box3d = encode_label(K, rot_y,
                                       np.array([length, height, width]), locs)

        if np.all(box2d == 0):
            continue

        imgpts, _ = cv2.projectPoints(box3d.T, rvec, tvec, K, 0)
        imgpts_list.append(imgpts)

    return imgpts_list


def draw_smoke_3d(img, imgpts_list):
    """draw smoke result to photo"""
    connect_line_id = [
        [1, 0],
        [2, 7],
        [3, 6],
        [4, 5],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [0, 7],
        [7, 6],
        [6, 5],
        [5, 0],
    ]

    img_draw = img.copy()

    for imgpts in imgpts_list:
        for p in imgpts:
            p_x, p_y = int(p[0][0]), int(p[0][1])
            cv2.circle(img_draw, (p_x, p_y), 1, (0, 255, 0), -1)
        for i, line_id in enumerate(connect_line_id):

            p1 = (int(imgpts[line_id[0]][0][0]), int(imgpts[line_id[0]][0][1]))
            p2 = (int(imgpts[line_id[1]][0][0]), int(imgpts[line_id[1]][0][1]))

            if i <= 3:  # body
                color = (255, 0, 0)
            elif i <= 7:  # head
                color = (0, 0, 255)
            else:  # tail
                color = (255, 255, 0)

            cv2.line(img_draw, p1, p2, color, 1)

    return img_draw


if __name__ == "__main__":
    args = parse_args()
    pred = init_predictor(args)
    # Listed below are camera intrinsic parameter of the kitti dataset
    # If the model is trained on other datasets, please replace the relevant data
    K = np.array(
        [[
            [721.53771973, 0.0, 609.55932617],
            [0.0, 721.53771973, 172.85400391],
            [0, 0, 1],
        ]],
        np.float32,
    )

    img, ori_img_size, output_size = get_img(args.image)
    ratio = get_ratio(ori_img_size, output_size)

    results = run(pred, img, K, ratio)

    total_pred = results[0]

    # convert pred to bboxes_2d, bboxes_3d
    kitti_records = total_pred_by_conf_to_kitti_records(total_pred, conf=0.5)
    bboxes_2d, bboxes_3d, labels = camera_record_to_object(kitti_records)
    # read origin image
    img_origin = cv2.imread(args.image)
    # to 8 points on image
    imgpts_list = make_imgpts_list(bboxes_3d, K[0])
    # draw smoke result to photo
    img_draw = draw_smoke_3d(img_origin, imgpts_list)
    cv2.imwrite("output.bmp", img_draw)
