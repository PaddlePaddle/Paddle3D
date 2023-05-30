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
import numpy as np
import cv2

from paddle3d.datasets.kitti.kitti_utils import camera_record_to_object

from demo.visualization_demo.vis_utils import Calibration, show_lidar_with_boxes, total_imgpred_by_conf_to_kitti_records, \
    make_imgpts_list, draw_mono_3d, show_bev_with_boxes

pth = '../datasets/KITTI/training'  # Kitti dataset path

files = os.listdir(os.path.join(pth, 'image_2'))
files = sorted(files)

mode = 'bev'

assert mode in ['bev', 'image', 'pcd'], ''

for img in files:
    id = img[:-4]
    label_file = os.path.join(pth, 'label_2', f'{id}.txt')
    calib_file = os.path.join(pth, 'calib', f'{id}.txt')
    img_file = os.path.join(pth, 'image_2', f'{id}.png')
    pcd_file = os.path.join(pth, 'velodyne', f'{id}.bin')

    label_lines = open(label_file).readlines()
    kitti_records_list = [line.strip().split(' ') for line in label_lines]

    if mode == 'pcd':
        box3d_list = []
        for itm in kitti_records_list:
            itm = [float(i) for i in itm[8:]]
            # [z, -x, -y, w, l, h, ry]
            box3d_list.append(
                [itm[5], -itm[3], -itm[4], itm[1], itm[2], itm[0], itm[6]])
        box3d = np.asarray(box3d_list)
        scan = np.fromfile(pcd_file, dtype=np.float32)
        pc_velo = scan.reshape((-1, 4))
        # Obtain calibration information about Kitti
        calib = Calibration(calib_file)
        # Plot box in lidar cloud
        # show_lidar_with_boxes(pc_velo, result['bboxes_3d'], result['confidences'], calib)
        show_lidar_with_boxes(pc_velo, box3d, np.ones(box3d.shape[0]), calib)

    if mode == 'image':
        kitti_records = np.array(kitti_records_list)
        bboxes_2d, bboxes_3d, labels = camera_record_to_object(kitti_records)
        # read origin image
        img_origin = cv2.imread(img_file)
        # to 8 points on image
        itms = open(calib_file).readlines()[2]
        P2 = itms[4:].strip().split(' ')
        K = np.asarray([float(i) for i in P2]).reshape(3, 4)[:, :3]
        imgpts_list = make_imgpts_list(bboxes_3d, K)
        # draw smoke result to photo
        draw_mono_3d(img_origin, imgpts_list)

    if mode == 'bev':
        box3d_list = []
        for itm in kitti_records_list:
            itm = [float(i) for i in itm[8:]]
            # [z, -x, -y, w, l, h, ry]
            box3d_list.append(
                [itm[5], -itm[3], -itm[4], itm[1], itm[2], itm[0], itm[6]])
        box3d = np.asarray(box3d_list)
        scan = np.fromfile(pcd_file, dtype=np.float32)
        pc_velo = scan.reshape((-1, 4))
        # Obtain calibration information about Kitti
        calib = Calibration(calib_file)
        # Plot box in lidar cloud (bev)
        show_bev_with_boxes(pc_velo, box3d, np.ones(box3d.shape[0]), calib)
