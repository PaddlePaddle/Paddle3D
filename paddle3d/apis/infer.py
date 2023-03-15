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

import copy
import os
import cv2
import sys
import numpy as np
from collections import defaultdict
from typing import Callable, Optional, Union

import paddle
from visualdl import LogWriter

import paddle3d.env as env
from paddle3d.apis.checkpoint import Checkpoint, CheckpointABC
from paddle3d.apis.pipeline import training_step, validation_step
from paddle3d.apis.scheduler import Scheduler, SchedulerABC
from paddle3d.utils.logger import logger
from paddle3d.utils.shm_utils import _get_shared_memory_size_in_M
from paddle3d.utils.timer import Timer

from paddle3d.datasets.kitti.kitti_utils import camera_record_to_object

from paddle3d.apis.trainer import Trainer

from demo.utils import Calibration, show_lidar_with_boxes, total_imgpred_by_conf_to_kitti_records, \
    make_imgpts_list, draw_mono_3d, show_bev_with_boxes


class Infer(Trainer):
    """
    """

    def __init__(
            self,
            model: paddle.nn.Layer,
            optimizer: paddle.optimizer.Optimizer,
            iters: Optional[int] = None,
            epochs: Optional[int] = None,
            train_dataset: Optional[paddle.io.Dataset] = None,
            val_dataset: Optional[paddle.io.Dataset] = None,
            resume: bool = False,
            # TODO: Default parameters should not use mutable objects, there is a risk
            checkpoint: Union[dict, CheckpointABC] = dict(),
            scheduler: Union[dict, SchedulerABC] = dict(),
            dataloader_fn: Union[dict, Callable] = dict(),
            amp_cfg: Optional[dict] = None):
        super(Infer,
              self).__init__(model, optimizer, iters, epochs, train_dataset,
                             val_dataset, resume, checkpoint, scheduler,
                             dataloader_fn, amp_cfg)

    def infer(self, mode) -> float:
        """
        """
        sync_bn = (getattr(self.model, 'sync_bn', False) and env.nranks > 1)
        if sync_bn:
            sparse_conv = False
            for layer in self.model.sublayers():
                if 'sparse' in str(type(layer)):
                    sparse_conv = True
                    break
            if sparse_conv:
                self.model = paddle.sparse.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.model)
            else:
                self.model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.model)

        if self.val_dataset is None:
            raise RuntimeError('No evaluation dataset specified!')
        msg = 'evaluate on validate dataset'
        metric_obj = self.val_dataset.metric

        for idx, sample in logger.enumerate(self.eval_dataloader, msg=msg):
            results = validation_step(self.model, sample)

            if mode == 'pcd':
                for result in results:
                    scan = np.fromfile(result['path'], dtype=np.float32)
                    pc_velo = scan.reshape((-1, 4))
                    # Obtain calibration information about Kitti
                    calib = Calibration(result['path'].replace(
                        'velodyne', 'calib').replace('bin', 'txt'))
                    # Plot box in lidar cloud
                    # show_lidar_with_boxes(pc_velo, result['bboxes_3d'], result['confidences'], calib)
                    show_lidar_with_boxes(pc_velo, result['bboxes_3d'],
                                          result['confidences'], calib)

            if mode == 'image':
                for result in results:
                    kitti_records = total_imgpred_by_conf_to_kitti_records(
                        result, 0.3)
                    bboxes_2d, bboxes_3d, labels = camera_record_to_object(
                        kitti_records)
                    # read origin image
                    img_origin = cv2.imread(result['path'])
                    # to 8 points on image
                    K = np.array(result['meta']['camera_intrinsic'])
                    imgpts_list = make_imgpts_list(bboxes_3d, K)
                    # draw smoke result to photo
                    draw_mono_3d(img_origin, imgpts_list)

            if mode == 'bev':
                for result in results:
                    scan = np.fromfile(result['path'], dtype=np.float32)
                    pc_velo = scan.reshape((-1, 4))
                    # Obtain calibration information about Kitti
                    calib = Calibration(result['path'].replace(
                        'velodyne', 'calib').replace('bin', 'txt'))
                    # Plot box in lidar cloud (bev)
                    show_bev_with_boxes(pc_velo, result['bboxes_3d'],
                                        result['confidences'], calib)
