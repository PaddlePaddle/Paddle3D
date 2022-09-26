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

import concurrent.futures as futures
import copy
import os
import pickle
from collections import defaultdict

import numpy as np
import paddle
import skimage.transform
from skimage import io

import paddle3d.transforms as T
from paddle3d.apis import manager
from paddle3d.datasets.kitti.kitti_det import KittiDetDataset
from paddle3d.datasets.kitti.kitti_metric import KittiDepthMetric
from paddle3d.datasets.kitti.kitti_utils import (Calibration,
                                                 get_objects_from_label)
from paddle3d.geometries.bbox import (boxes3d_kitti_camera_to_lidar,
                                      boxes_to_corners_3d, in_hull,
                                      mask_boxes_outside_range_numpy)
from paddle3d.thirdparty import kitti_eval


def get_pad_params(desired_size, cur_size):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/common_utils.py#L112

    Get padding parameters for np.pad function
    Args:
        desired_size [int]: Desired padded output size
        cur_size [int]: Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params [tuple(int)]: Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


@manager.DATASETS.add_component
class KittiDepthDataset(KittiDetDataset):
    """
    This code is based on https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/datasets/kitti/kitti_dataset.py#L17
    """

    def __init__(self,
                 dataset_root,
                 mode,
                 point_cloud_range,
                 depth_downsample_factor,
                 voxel_size,
                 class_names,
                 remove_outside_boxes=True):
        super(KittiDepthDataset, self).__init__(dataset_root, mode)
        self.class_names = class_names
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.depth_downsample_factor = depth_downsample_factor
        self._merge_all_iters_to_one_epoch = False
        self.remove_outside_boxes = remove_outside_boxes
        self.voxel_size = voxel_size
        self.grid_size = None
        self.training = mode == 'train'
        self.kitti_infos = []
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        kitti_infos = []

        info_path = os.path.join(self.dataset_root,
                                 "kitti_infos_" + mode + '.pkl')
        if not os.path.exists(info_path):
            return
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)
        self.calculate_grid_size()

    def set_split(self, split):
        split_dir = os.path.join(self.dataset_root, 'ImageSets', split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()
                               ] if os.path.exists(split_dir) else None

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.base_dir, 'velodyne', '%s.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx [int]: Index of the image sample
        Returns:
            image [np.ndarray(H, W, 3)]: RGB Image
        """
        img_file = os.path.join(self.base_dir, 'image_2', '%s.png' % idx)
        assert os.path.exists(img_file)
        image = io.imread(img_file)
        image = image[:, :, :3]
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = os.path.join(self.base_dir, 'image_2', '%s.png' % idx)
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = os.path.join(self.base_dir, 'label_2', '%s.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx [str]: Index of the sample
        Returns:
            depth [np.ndarray(H, W)]: Depth map
        """
        depth_file = os.path.join(self.base_dir, 'depth_2', '%s.png' % idx)
        assert os.path.exists(depth_file)
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        depth = skimage.transform.downscale_local_mean(
            image=depth,
            factors=(self.depth_downsample_factor,
                     self.depth_downsample_factor))
        return depth

    def get_calib(self, idx):
        _, _, P2, _, R0_rect, V2C, _ = self.load_calibration_info(
            idx, use_data=False)
        calib_dict = {"P2": P2, "R0": R0_rect, "Tr_velo2cam": V2C}
        return Calibration(calib_dict)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.base_dir, 'planes', '%s.txt' % idx)
        if not os.path.exists(plane_file):
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0,
                                    pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0,
                                    pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self,
                  num_workers=4,
                  has_label=True,
                  count_inside_pts=True,
                  sample_id_list=None,
                  mode='train'):
        def process_single_scene(sample_idx):
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {
                'image_idx': sample_idx,
                'image_shape': self.get_image_shape(sample_idx)
            }
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate(
                [calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate(
                [calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {
                'P2': P2,
                'R0_rect': R0_4x4,
                'Tr_velo_to_cam': V2C_4x4
            }

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array(
                    [obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array(
                    [obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array(
                    [obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate(
                    [obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array(
                    [[obj.l, obj.h, obj.w]
                     for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate(
                    [obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array(
                    [obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array(
                    [obj.level for obj in obj_list], np.int32)

                num_objects = len([
                    obj.cls_type for obj in obj_list
                    if obj.cls_type != 'DontCare'
                ])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate(
                    [loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])],
                    axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

            return info

        self.mode = mode
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def update_data(self, data_dict):
        """
        Updates data dictionary with additional items
        Args:
            data_dict [dict]: Data dictionary returned by __getitem__
        Returns:
            data_dict [dict]: Updated data dictionary returned by __getitem__
        """
        # Image
        data_dict['images'] = self.get_image(data_dict["frame_id"])

        # Depth Map
        data_dict['depth_maps'] = self.get_depth_map(data_dict["frame_id"])

        # Calibration matricies
        # Convert calibration matrices to homogeneous format and combine
        calib = data_dict["calib"]
        V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1],
                                             dtype=np.float32)))  # (4, 4)
        R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        V2R = R0 @ V2C
        data_dict.update({
            "trans_lidar_to_cam": V2R,
            "trans_cam_to_img": calib.P2,
            "R0": calib.R0,
            "Tr_velo2cam": calib.V2C
        })
        return data_dict

    @property
    def metric(self, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}
        eval_gt_annos = [
            copy.deepcopy(info['annos']) for info in self.kitti_infos
        ]
        return KittiDepthMetric(
            eval_gt_annos=eval_gt_annos, class_names=self.class_names)

    def mask_points_and_boxes_outside_range(self, data_dict):
        if data_dict.get(
                'gt_boxes', None
        ) is not None and self.remove_outside_boxes and self.training:
            mask = mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'],
                self.point_cloud_range,
                min_num_corners=1)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def calculate_grid_size(self):
        grid_size = (self.point_cloud_range[3:6] -
                     self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

    def drop_info_with_name(self, info, name):
        ret_info = {}
        keep_indices = [i for i, x in enumerate(info['name']) if x != name]
        for key in info.keys():
            ret_info[key] = info[key][keep_indices]
        return ret_info

    def data_augmentor(self, data_dict):
        data_dict = T.functional.random_depth_image_horizontal(data_dict)

        data_dict['gt_boxes'][:, 6] = data_dict['gt_boxes'][:, 6] - np.floor(
            data_dict['gt_boxes'][:, 6] / (2 * np.pi) + 0.5) * (2 * np.pi)
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')

        return data_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...
        """
        if data_dict.get('gt_boxes', None) is not None:
            selected = [
                i for i, x in enumerate(data_dict['gt_names'])
                if x in self.class_names
            ]
            selected = np.array(selected, dtype=np.int64)
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in data_dict['gt_names']],
                dtype=np.int32)
            gt_classes = gt_classes.reshape(-1, 1).astype(np.float32)

            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes),
                                      axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_box2d', None) is not None:
                data_dict['gt_box2d'] = data_dict['gt_box2d'][selected]
                gt_boxes_2d = np.concatenate(
                    (data_dict['gt_box2d'], gt_classes), axis=1)
                data_dict['gt_box2d'] = gt_boxes_2d

        if data_dict is not None:
            data_dict = self.mask_points_and_boxes_outside_range(
                data_dict=data_dict)

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        data_dict.pop('calib', None)
        data_dict.pop('frame_id', None)

        return data_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        calib = self.get_calib(sample_idx)
        img_shape = info['image']['image_shape']
        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
            'calib_info': info['calib'],
            'image_shape': img_shape
        }

        if 'annos' in info:
            annos = info['annos']
            annos = self.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos[
                'rotation_y']
            gt_names = annos['name']
            bbox = annos['bbox']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                             axis=1).astype(np.float32)
            gt_boxes_lidar = boxes3d_kitti_camera_to_lidar(
                gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
                'gt_boxes2d': bbox
            })

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.update_data(data_dict=input_dict)
        data_dict = self.prepare_data(data_dict=data_dict)
        data_dict['image_shape'] = img_shape
        return data_dict

    @staticmethod
    def collate_fn(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, max_gt, val[0].shape[-1]),
                        dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros(
                        (batch_size, max_boxes, val[0].shape[-1]),
                        dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = get_pad_params(
                            desired_size=max_h, cur_size=image.shape[0])
                        pad_w = get_pad_params(
                            desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = 0

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(
                            image,
                            pad_width=pad_width,
                            mode='constant',
                            constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                    if key == "images":
                        ret[key] = ret[key].transpose([0, 3, 1, 2])
                elif key in "calib_info":
                    continue
                else:
                    ret[key] = np.stack(val, axis=0)

            except:
                raise TypeError("Error in collate_batch: key={}.".format(key))

        ret['batch_size'] = batch_size
        return ret
