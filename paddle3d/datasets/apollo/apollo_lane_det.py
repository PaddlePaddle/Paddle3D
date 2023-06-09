# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# ------------------------------------------------------------------------
# Modified from BEV-LaneDet (https://github.com/gigo-team/bev_lane_det)
# ------------------------------------------------------------------------

import os
import cv2
import copy
import json
import numpy as np
from scipy.interpolate import interp1d

import paddle
from paddle3d.apis import manager
import paddle3d.transforms as T
import albumentations as A

from .apollo_lane_metric import ApolloLaneMetric
from .standard_camera_cpu import Standard_camera
from .coord_util import ego2image, IPM2ego_matrix


@manager.DATASETS.add_component
class ApolloOffsetDataset(paddle.io.Dataset):
    def __init__(
            self,
            data_json_path,
            dataset_base_dir,
            x_range,
            y_range,
            meter_per_pixel,
            input_shape,
            output_2d_shape,
            virtual_camera_config,
            transforms=None,
    ):

        self.x_range = x_range
        self.y_range = y_range
        self.meter_per_pixel = meter_per_pixel
        self.cnt_list = []
        self.lane3d_thick = 1
        self.lane2d_thick = 3
        json_file_path = data_json_path
        self.dataset_base_dir = dataset_base_dir
        with open(json_file_path, 'r') as file:
            for line in file:
                info_dict = json.loads(line)
                self.cnt_list.append(info_dict)
        ''' virtual camera paramter'''
        self.use_virtual_camera = virtual_camera_config['use_virtual_camera']

        camera_ext_virtual, camera_K_virtual = get_camera_matrix(
            0.04325083977888603, 1.7860000133514404)  # a random parameter
        self.vc_intrinsic = camera_K_virtual
        self.vc_extrinsics = np.linalg.inv(camera_ext_virtual)
        self.vc_image_shape = tuple(virtual_camera_config['vc_image_shape'])
        ''' transform loader '''
        self.output2d_size = output_2d_shape
        if transforms is None:
            self.trans_image = A.Compose([
                A.Resize(height=input_shape[0], width=input_shape[1]),
                A.MotionBlur(p=0.2),
                A.RandomBrightnessContrast(),
                A.ColorJitter(p=0.1),
                A.Normalize(),
            ])
        self.ipm_h, self.ipm_w = int(
            (self.x_range[1] - self.x_range[0]) / self.meter_per_pixel), int(
                (self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)

        self.is_train_mode = True

    def get_y_offset_and_z(self, res_d):
        def caculate_distance(base_points, lane_points, lane_z,
                              lane_points_set):
            condition = np.where((lane_points_set[0] == int(base_points[0])) &
                                 (lane_points_set[1] == int(base_points[1])))
            if len(condition[0]) == 0:
                return None, None
            lane_points_selected = lane_points.T[condition]  # 找到bin
            lane_z_selected = lane_z.T[condition]
            offset_y = np.mean(lane_points_selected[:, 1]) - base_points[1]
            z = np.mean(lane_z_selected[:, 1])
            return offset_y, z

        res_lane_points = {}
        res_lane_points_z = {}
        res_lane_points_bin = {}
        res_lane_points_set = {}
        for idx in res_d:
            ipm_points_ = np.array(res_d[idx])
            ipm_points = ipm_points_.T[np.where(
                (ipm_points_[1] >= 0) & (ipm_points_[1] < self.ipm_h))].T
            if len(ipm_points[0]) <= 1:
                continue
            x, y, z = ipm_points[1], ipm_points[0], ipm_points[2]
            base_points = np.linspace(x.min(), x.max(),
                                      int((x.max() - x.min()) // 0.05))
            base_points_bin = np.linspace(
                int(x.min()), int(x.max()),
                int(int(x.max()) - int(x.min())) + 1)

            if len(x) == len(set(x)):
                if len(x) <= 1:
                    continue
                elif len(x) <= 2:
                    function1 = interp1d(
                        x, y, kind='linear', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) <= 3:
                    function1 = interp1d(
                        x, y, kind='quadratic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    function1 = interp1d(
                        x, y, kind='cubic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='cubic')
            else:
                sorted_index = np.argsort(x)[::-1]
                x_, y_, z_ = [], [], []
                for x_index in range(len(sorted_index)):
                    if x[sorted_index[x_index]] >= x[sorted_index[
                            x_index - 1]] and x_index != 0:
                        continue
                    else:
                        x_.append(x[sorted_index[x_index]])
                        y_.append(y[sorted_index[x_index]])
                        z_.append(z[sorted_index[x_index]])
                x, y, z = np.array(x_), np.array(y_), np.array(z_)
                if len(x) <= 1:
                    continue
                elif len(x) <= 2:
                    function1 = interp1d(
                        x, y, kind='linear', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) <= 3:
                    function1 = interp1d(
                        x, y, kind='quadratic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    function1 = interp1d(
                        x, y, kind='cubic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='cubic')

            y_points = function1(base_points)
            y_points_bin = function1(base_points_bin)
            z_points = function2(base_points)

            res_lane_points[idx] = np.array([base_points, y_points])
            res_lane_points_z[idx] = np.array([base_points, z_points])
            res_lane_points_bin[idx] = np.array([base_points_bin,
                                                 y_points_bin]).astype(np.int)
            res_lane_points_set[idx] = np.array([base_points,
                                                 y_points]).astype(np.int)
        offset_map = np.zeros((self.ipm_h, self.ipm_w))
        z_map = np.zeros((self.ipm_h, self.ipm_w))
        ipm_image = np.zeros((self.ipm_h, self.ipm_w))
        for idx in res_lane_points_bin:
            lane_bin = res_lane_points_bin[idx].T
            for point in lane_bin:
                row, col = point[0], point[1]
                if not (0 < row < self.ipm_h and 0 < col < self.ipm_w):
                    continue
                ipm_image[row, col] = idx
                center = np.array([row, col])
                offset_y, z = caculate_distance(center, res_lane_points[idx],
                                                res_lane_points_z[idx],
                                                res_lane_points_set[idx])
                if offset_y is None:
                    ipm_image[row, col] = 0
                    continue
                if offset_y > 1:
                    offset_y = 1
                if offset_y < 0:
                    offset_y = 0
                offset_map[row][col] = offset_y
                z_map[row][col] = z

        return ipm_image, offset_map, z_map

    def get_seg_offset(self, idx):
        info_dict = self.cnt_list[idx]
        name_list = info_dict['raw_file'].split('/')
        image_path = os.path.join(self.dataset_base_dir, 'images',
                                  name_list[-2], name_list[-1])
        image = cv2.imread(image_path)

        # caculate camera parameter
        cam_height, cam_pitch = info_dict['cam_height'], info_dict['cam_pitch']
        project_g2c, camera_k = self.get_camera_matrix(cam_pitch, cam_height)
        project_c2g = np.linalg.inv(project_g2c)

        # caculate point
        lane_grounds = info_dict['laneLines']
        image_gt = np.zeros(image.shape[:2], dtype=np.uint8)
        matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel),
                        int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel)
        res_points_d = {}
        for lane_idx in range(len(lane_grounds)):
            # select point by visibility
            lane_visibility = np.array(
                info_dict['laneLines_visibility'][lane_idx])
            lane_ground = np.array(lane_grounds[lane_idx])
            assert lane_visibility.shape[0] == lane_ground.shape[0]
            lane_ground = lane_ground[lane_visibility > 0.5]
            lane_ground = np.concatenate(
                [lane_ground, np.ones([lane_ground.shape[0], 1])], axis=1).T
            # get image gt
            lane_camera = np.matmul(project_g2c, lane_ground)
            lane_image = camera_k @ lane_camera[:3]
            lane_image = lane_image / lane_image[2]
            lane_uv = lane_image[:2].T
            cv2.polylines(image_gt, [lane_uv.astype(np.int)], False,
                          lane_idx + 1, 3)
            x, y, z = lane_ground[1], -1 * lane_ground[0], lane_ground[2]
            ground_points = np.array([x, y])
            ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (
                ground_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
            ipm_points_ = np.zeros_like(ipm_points)
            ipm_points_[0] = ipm_points[1]
            ipm_points_[1] = ipm_points[0]
            res_points = np.concatenate([ipm_points_, np.array([z])], axis=0)
            res_points_d[lane_idx + 1] = res_points

        bev_gt, offset_y_map, z_map = self.get_y_offset_and_z(res_points_d)
        ''' virtual camera '''
        if self.use_virtual_camera:
            sc = Standard_camera(
                self.vc_intrinsic, self.vc_extrinsics,
                (self.vc_image_shape[1], self.vc_image_shape[0]), camera_k,
                project_c2g, image.shape[:2])
            trans_matrix = sc.get_matrix(height=0)
            image = cv2.warpPerspective(image, trans_matrix,
                                        self.vc_image_shape)
            image_gt = cv2.warpPerspective(image_gt, trans_matrix,
                                           self.vc_image_shape)
        return image, image_gt, bev_gt, offset_y_map, z_map, project_c2g, camera_k

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        image, image_gt, bev_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic = self.get_seg_offset(
            idx)
        transformed = self.trans_image(image=image)
        image = transformed["image"]

        image = image.transpose([2, 0, 1])
        ''' 2d gt'''
        image_gt = cv2.resize(
            image_gt, (self.output2d_size[1], self.output2d_size[0]),
            interpolation=cv2.INTER_NEAREST)
        image_gt_instance = np.array(
            image_gt, dtype=np.float32)[np.newaxis, ...]  # h, w, c
        image_gt_segment = copy.deepcopy(image_gt_instance)

        image_gt_segment[image_gt_segment > 0] = 1
        ''' 3d gt'''
        bev_gt_instance = np.array(
            bev_gt, dtype=np.float32)[np.newaxis, ...]  # h, w, c0
        bev_gt_offset = np.array(
            offset_y_map, dtype=np.float32)[np.newaxis, ...]
        bev_gt_z = np.array(z_map, dtype=np.float32)[np.newaxis, ...]
        bev_gt_segment = copy.deepcopy(bev_gt_instance)
        bev_gt_segment[bev_gt_segment > 0] = 1

        sample = dict(
            img=image,
            bev_gt_segment=bev_gt_segment,
            bev_gt_instance=bev_gt_instance,
            bev_gt_offset=bev_gt_offset,
            bev_gt_z=bev_gt_z,
            image_gt_segment=image_gt_segment,
            image_gt_instance=image_gt_instance)
        return sample

    def get_camera_matrix(self, cam_pitch, cam_height):
        proj_g2c = np.array([[1, 0, 0, 0],
                             [
                                 0,
                                 np.cos(np.pi / 2 + cam_pitch),
                                 -np.sin(np.pi / 2 + cam_pitch), cam_height
                             ],
                             [
                                 0,
                                 np.sin(np.pi / 2 + cam_pitch),
                                 np.cos(np.pi / 2 + cam_pitch), 0
                             ], [0, 0, 0, 1]])

        camera_K = np.array([[2015., 0., 960.], [0., 2015., 540.], [0., 0.,
                                                                    1.]])

        return proj_g2c, camera_K

    def __len__(self):
        return len(self.cnt_list)


@manager.DATASETS.add_component
class ApolloOffsetValDataset(paddle.io.Dataset):
    def __init__(self,
                 data_json_path,
                 dataset_base_dir,
                 virtual_camera_config,
                 x_range,
                 meter_per_pixel,
                 transforms=None):

        if isinstance(transforms, list):
            transforms = paddle.vision.Compose(transforms)

        self.cnt_list = []
        self.data_json_path = data_json_path
        json_file_path = data_json_path
        self.dataset_base_dir = dataset_base_dir
        self.x_range = x_range
        self.meter_per_pixel = meter_per_pixel

        with open(json_file_path, 'r') as file:
            for line in file:
                info_dict = json.loads(line)
                self.cnt_list.append(info_dict)
        ''' virtual camera paramter'''
        self.use_virtual_camera = virtual_camera_config['use_virtual_camera']
        camera_ext_virtual, camera_K_virtual = get_camera_matrix(
            0.04325083977888603, 1.7860000133514404)  # a random parameter
        self.vc_intrinsic = camera_K_virtual
        self.vc_extrinsics = np.linalg.inv(camera_ext_virtual)

        self.vc_image_shape = tuple(virtual_camera_config['vc_image_shape'])
        ''' transform loader '''
        self.trans_image = transforms

        self.is_train_mode = False

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''

        info_dict = self.cnt_list[idx]
        name_list = info_dict['raw_file'].split('/')
        image_path = os.path.join(self.dataset_base_dir, 'images',
                                  name_list[-2], name_list[-1])
        image = cv2.imread(image_path)

        # caculate camera parameter
        cam_height, cam_pitch = info_dict['cam_height'], info_dict['cam_pitch']
        project_g2c, camera_k = self.get_camera_matrix(cam_pitch, cam_height)
        project_c2g = np.linalg.inv(project_g2c)
        ''' virtual camera '''
        if self.use_virtual_camera:
            sc = Standard_camera(
                self.vc_intrinsic, self.vc_extrinsics,
                (self.vc_image_shape[1], self.vc_image_shape[0]), camera_k,
                project_c2g, image.shape[:2])
            trans_matrix = sc.get_matrix(height=0)
            image = cv2.warpPerspective(image, trans_matrix,
                                        self.vc_image_shape)

        image = self.trans_image(image)

        return dict(img=image, name_list=name_list[1:])

    def get_camera_matrix(self, cam_pitch, cam_height):
        proj_g2c = np.array([[1, 0, 0, 0],
                             [
                                 0,
                                 np.cos(np.pi / 2 + cam_pitch),
                                 -np.sin(np.pi / 2 + cam_pitch), cam_height
                             ],
                             [
                                 0,
                                 np.sin(np.pi / 2 + cam_pitch),
                                 np.cos(np.pi / 2 + cam_pitch), 0
                             ], [0, 0, 0, 1]])

        camera_K = np.array([[2015., 0., 960.], [0., 2015., 540.], [0., 0.,
                                                                    1.]])

        return proj_g2c, camera_K

    def __len__(self):
        return len(self.cnt_list)

    @property
    def metric(self):
        return ApolloLaneMetric(self.data_json_path, self.x_range,
                                self.meter_per_pixel)


def get_camera_matrix(cam_pitch, cam_height):
    proj_g2c = np.array(
        [[1, 0, 0, 0],
         [
             0,
             np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch),
             cam_height
         ],
         [0, np.sin(np.pi / 2 + cam_pitch),
          np.cos(np.pi / 2 + cam_pitch), 0], [0, 0, 0, 1]])

    camera_K = np.array([[2015., 0., 960.], [0., 2015., 540.], [0., 0., 1.]])

    return proj_g2c, camera_K
