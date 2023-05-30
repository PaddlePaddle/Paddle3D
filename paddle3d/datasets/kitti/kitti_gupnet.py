# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from PIL import Image
from typing import List, Dict
from paddle3d.datasets.kitti.kitti_det import KittiDetDataset
from paddle3d.datasets.kitti.kitti_utils import Object3d, Calibration, box_lidar_to_camera, filter_fake_result, camera_record_to_object
from paddle3d.datasets.kitti.kitti_gupnet_utils import get_affine_transform, affine_transform, gaussian_radius, draw_umich_gaussian, angle2class
from paddle3d.apis import manager
from paddle3d.sample import Sample
from paddle3d.datasets.metrics import MetricABC
from paddle3d.geometries.bbox import (BBoxes2D, BBoxes3D, CoordMode,
                                      project_to_image)
from paddle3d.thirdparty import kitti_eval
from paddle3d.utils.logger import logger


@manager.DATASETS.add_component
class GUPKittiMonoDataset(KittiDetDataset):
    """
    """

    def __init__(self,
                 dataset_root,
                 use_3d_center=True,
                 class_name=['Pedestrian', 'Car', 'Cyclist'],
                 resolution=[1280, 384],
                 random_flip=0.5,
                 random_crop=0.5,
                 scale=0.4,
                 shift=0.1,
                 mode='train'):
        super().__init__(dataset_root=dataset_root, mode=mode)
        self.dataset_root = dataset_root
        # basic configuration
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = class_name
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array(resolution)  # W * H
        self.use_3d_center = use_3d_center

        # l,w,h
        self.cls_mean_size = np.array(
            [[1.76255119, 0.66068622, 0.84422524],
             [1.52563191462, 1.62856739989, 3.88311640418],
             [1.73698127, 0.59706367, 1.76282397]])

        # data mode loading
        assert mode in ['train', 'val', 'trainval', 'test']
        self.mode = mode.lower()
        split_dir = os.path.join(dataset_root, 'ImageSets', mode + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.image_dir = os.path.join(self.base_dir, 'image_2')
        # data augmentation configuration
        self.data_augmentation = True if mode in ['train', 'trainval'
                                                  ] else False
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.scale = scale
        self.shift = shift

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        # print(img_file)
        assert os.path.exists(img_file)
        return Image.open(img_file)  # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return self.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        # assert os.path.exists(calib_file)
        if not os.path.exists(calib_file):
            print('Non-exist: ', calib_file)
        calib = self.get_calib_from_file(calib_file)
        return Calibration(calib)

    def get_objects_from_label(self, label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        objects = [Object3d(line) for line in lines]
        return objects

    def get_calib_from_file(self, calib_file):
        with open(calib_file) as f:
            lines = f.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        return {
            'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)
        }

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        # get samples
        filename = '{}.png'.format(self.data[item])
        path = os.path.join(self.image_dir, filename)
        calibs = self.load_calibration_info(item)

        sample = Sample(path=path, modality="image")
        # P2
        sample.meta.camera_intrinsic = calibs[2][:3, :3]
        sample.meta.id = self.data[item]
        sample.calibs = calibs

        kitti_records, _ = self.load_annotation(item)
        bboxes_2d, bboxes_3d, labels = camera_record_to_object(kitti_records)

        sample.bboxes_2d = bboxes_2d
        sample.bboxes_3d = bboxes_3d
        sample.labels = np.array([self.CLASS_MAP[label] for label in labels],
                                 dtype=np.int32)

        # get inputs
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        img_size = np.array(img.size)

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_flip_flag = False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                crop_size = img_size * \
                    np.clip(np.random.randn() * self.scale +
                            1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * \
                    np.clip(np.random.randn() * self.shift, -
                            2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * \
                    np.clip(np.random.randn() * self.shift, -
                            2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(
            center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(
            tuple(self.resolution.tolist()),
            method=Image.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.BILINEAR)
        coord_range = np.array([center - crop_size / 2,
                                center + crop_size / 2]).astype(np.float32)

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        # get calib
        calib = self.get_calib(index)

        features_size = self.resolution // self.downsample  # W * H
        #  ============================   get labels   ==============================
        if self.mode != 'test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0], object.box2d[2] = img_size[0] - \
                        x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.loc[0] *= -1
                    if object.ry > np.pi:
                        object.ry -= 2 * np.pi
                    if object.ry < -np.pi:
                        object.ry += 2 * np.pi
            # labels encoding
            heatmap = np.zeros(
                (self.num_classes, features_size[1], features_size[0]),
                dtype=np.float32)  # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            object_num = len(
                objects) if len(objects) < self.max_objs else self.max_objs
            for i in range(object_num):
                # filter objects by class_name
                if objects[i].cls_type not in self.class_name:
                    continue

                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].loc[-1] < 2:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample

                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2,
                                      (bbox_2d[1] + bbox_2d[3]) / 2],
                                     dtype=np.float32)  # W * H
                # real 3D center in 3D space
                center_3d = objects[i].loc + [0, -objects[i].h / 2, 0]
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                # project 3D center to image plane
                center_3d, _ = calib.rect_to_img(center_3d)
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(
                    np.int32) if self.use_3d_center else center_2d.astype(
                        np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[
                        0]:
                    continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[
                        1]:
                    continue

                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))

                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue

                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * \
                    features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h

                # encoding depth
                depth[i] = objects[i].loc[-1]

                # encoding heading angle
                # heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(
                    objects[i].ry,
                    (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
                if heading_angle > np.pi:
                    heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi:
                    heading_angle += 2 * np.pi
                # Convert continuous angle to discrete class and residual
                heading_bin[i], heading_res[i] = angle2class(heading_angle)

                # encoding 3d offset & size_3d
                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array(
                    [objects[i].h, objects[i].w, objects[i].l],
                    dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size

                # objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                if objects[i].truncation <= 0.5 and objects[i].occlusion <= 2:
                    mask_2d[i] = 1

            targets = {
                'depth': depth,
                'size_2d': size_2d,
                'heatmap': heatmap,
                'offset_2d': offset_2d,
                'indices': indices,
                'size_3d': size_3d,
                'offset_3d': offset_3d,
                'heading_bin': heading_bin,
                'heading_res': heading_res,
                'cls_ids': cls_ids,
                'mask_2d': mask_2d
            }
        else:
            targets = {}
        # collect return data
        inputs = img
        info = {
            'img_id': index,
            'img_size': img_size,
            'bbox_downsample_ratio': img_size / features_size
        }

        return inputs, calib.P2, coord_range, targets, info, sample

    @property
    def name(self) -> str:
        return "KITTI"

    @property
    def labels(self) -> List[str]:
        return self.class_name

    @property
    def metric(self):
        gt = []
        for idx in range(len(self)):
            annos = self.load_annotation(idx)
            if len(annos[0]) > 0 and len(annos[1]) > 0:
                gt.append(np.concatenate((annos[0], annos[1]), axis=0))
            elif len(annos[0]) > 0:
                gt.append(annos[0])
            else:
                gt.append(annos[1])
        return GUPKittiMetric(
            groundtruths=gt,
            classmap={i: name
                      for i, name in enumerate(self.class_names)},
            indexes=self.data)


class GUPKittiMetric(MetricABC):
    def __init__(self, groundtruths: List[np.ndarray], classmap: Dict[int, str],
                 indexes: List):
        self.gt_annos = groundtruths
        self.predictions = []
        self.classmap = classmap
        self.indexes = indexes

    def _parse_gt_to_eval_format(self,
                                 groundtruths: List[np.ndarray]) -> List[dict]:
        res = []
        for rows in groundtruths:
            if rows.size == 0:
                res.append({
                    'name': np.zeros([0]),
                    'truncated': np.zeros([0]),
                    'occluded': np.zeros([0]),
                    'alpha': np.zeros([0]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.zeros([0]),
                    'score': np.zeros([0])
                })
            else:
                res.append({
                    'name': rows[:, 0],
                    'truncated': rows[:, 1].astype(np.float64),
                    'occluded': rows[:, 2].astype(np.int64),
                    'alpha': rows[:, 3].astype(np.float64),
                    'bbox': rows[:, 4:8].astype(np.float64),
                    'dimensions': rows[:, [10, 8, 9]].astype(np.float64),
                    'location': rows[:, 11:14].astype(np.float64),
                    'rotation_y': rows[:, 14].astype(np.float64)
                })

        return res

    def get_camera_box2d(self, bboxes_3d: BBoxes3D, proj_mat: np.ndarray):
        box_corners = bboxes_3d.corners_3d
        box_corners_in_image = project_to_image(box_corners, proj_mat)
        minxy = np.min(box_corners_in_image, axis=1)
        maxxy = np.max(box_corners_in_image, axis=1)
        box_2d_preds = BBoxes2D(np.concatenate([minxy, maxxy], axis=1))

        return box_2d_preds

    def _parse_predictions_to_eval_format(
            self, predictions: List[Sample]) -> List[dict]:
        res = {}
        for pred in predictions:
            filter_fake_result(pred)
            id = pred.meta.id
            if pred.bboxes_3d is None:
                det = {
                    'truncated': np.zeros([0]),
                    'occluded': np.zeros([0]),
                    'alpha': np.zeros([0]),
                    'name': np.zeros([0]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.zeros([0]),
                    'score': np.zeros([0]),
                }
            else:
                num_boxes = pred.bboxes_3d.shape[0]
                names = np.array(
                    [self.classmap[label] for label in pred.labels])
                calibs = pred.calibs

                alpha = pred.get('alpha', np.zeros([num_boxes]))

                if pred.bboxes_3d.coordmode != CoordMode.KittiCamera:
                    bboxes_3d = box_lidar_to_camera(pred.bboxes_3d, calibs)
                else:
                    bboxes_3d = pred.bboxes_3d

                if bboxes_3d.origin != [.5, 1., .5]:
                    bboxes_3d[:, :3] += bboxes_3d[:, 3:6] * (
                        np.array([.5, 1., .5]) - np.array(bboxes_3d.origin))
                    bboxes_3d.origin = [.5, 1., .5]

                if pred.bboxes_2d is None:
                    bboxes_2d = self.get_camera_box2d(bboxes_3d, calibs[2])
                else:
                    bboxes_2d = pred.bboxes_2d

                loc = bboxes_3d[:, :3]
                dim = bboxes_3d[:, 3:6]

                det = {
                    # fake value
                    'truncated': np.zeros([num_boxes]),
                    'occluded': np.zeros([num_boxes]),
                    # predict value
                    'alpha': alpha,
                    'name': names,
                    'bbox': bboxes_2d,
                    'dimensions': dim,
                    # TODO: coord trans
                    'location': loc,
                    'rotation_y': bboxes_3d[:, 6],
                    'score': pred.confidences,
                }

            res[id] = det

        return [res[idx] for idx in self.indexes]

    def update(self, predictions: List[Sample], **kwargs):
        """
        """
        self.predictions += predictions

    def compute(self, verbose=False, **kwargs) -> dict:
        """
        """
        gt_annos = self._parse_gt_to_eval_format(self.gt_annos)
        dt_annos = self._parse_predictions_to_eval_format(self.predictions)

        if len(dt_annos) != len(gt_annos):
            raise RuntimeError(
                'The number of predictions({}) is not equal to the number of GroundTruths({})'
                .format(len(dt_annos), len(gt_annos)))

        metric_r40_dict = kitti_eval(
            gt_annos,
            dt_annos,
            current_classes=list(self.classmap.values()),
            metric_types=["bbox", "bev", "3d"],
            recall_type='R40')

        metric_r11_dict = kitti_eval(
            gt_annos,
            dt_annos,
            current_classes=list(self.classmap.values()),
            metric_types=["bbox", "bev", "3d"],
            recall_type='R11')

        if verbose:
            for cls, cls_metrics in metric_r40_dict.items():
                logger.info("{}:".format(cls))
                for overlap_thresh, metrics in cls_metrics.items():
                    for metric_type, thresh in zip(["bbox", "bev", "3d"],
                                                   overlap_thresh):
                        if metric_type in metrics:
                            logger.info(
                                "{} AP_R40@{:.0%}: {:.2f} {:.2f} {:.2f}".format(
                                    metric_type.upper().ljust(4), thresh,
                                    *metrics[metric_type]))

            for cls, cls_metrics in metric_r11_dict.items():
                logger.info("{}:".format(cls))
                for overlap_thresh, metrics in cls_metrics.items():
                    for metric_type, thresh in zip(["bbox", "bev", "3d"],
                                                   overlap_thresh):
                        if metric_type in metrics:
                            logger.info(
                                "{} AP_R11@{:.0%}: {:.2f} {:.2f} {:.2f}".format(
                                    metric_type.upper().ljust(4), thresh,
                                    *metrics[metric_type]))
        return metric_r40_dict, metric_r11_dict
