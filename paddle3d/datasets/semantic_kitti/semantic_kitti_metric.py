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

from typing import Dict, List

import numpy as np
import paddle

from paddle3d.datasets.metrics import MetricABC
from paddle3d.sample import Sample
from paddle3d.utils.logger import logger

from .semantic_kitti import SemanticKITTIDataset

__all__ = ["SemanticKITTIMetric"]


class SemanticKITTIMetric(MetricABC):
    """
    IoU evaluation of semantic segmentation task on SemanticKITTI dataset, with Paddle as backend.
    Please refer to:
    <https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/np_ioueval.py>.

    Args:
        num_classes (int): The number of classes.
        ignore (List[int]): Classes indices that are ignored during evaluation.
    """

    def __init__(self, num_classes: int, ignore: List[int] = None):
        # classes
        self.num_classes = num_classes

        # What to include and ignore from the means
        include = [n for n in range(self.num_classes) if n not in ignore]
        logger.info("[IOU EVAL] IGNORED CLASSES: {}".format(ignore))
        logger.info("[IOU EVAL] INCLUDED CLASSES: {}".format(include))

        self.ignore = paddle.to_tensor(ignore, dtype="int64")
        self.include = paddle.to_tensor(
            [n for n in range(self.num_classes) if n not in ignore],
            dtype="int64")

        # reset the class counters
        self.reset()

    def num_classes(self):
        return self.num_classes

    def reset(self):
        self.conf_matrix = paddle.zeros((self.num_classes, self.num_classes),
                                        dtype="int64")

    def update(self, predictions: List[Sample],
               ground_truths: Dict):  # x=preds, y=targets
        for pd_sample, gt in zip(predictions, ground_truths["labels"]):
            pd = pd_sample.labels
            if isinstance(pd, np.ndarray):
                pd = paddle.to_tensor(pd, dtype="int64")
            if isinstance(gt, np.ndarray):
                gt = paddle.to_tensor(gt, dtype="int64")

            # sizes should be matching
            pd_row = pd.reshape([-1])  # de-batchify
            gt_row = gt.reshape([-1])  # de-batchify

            # check
            assert (pd_row.shape == gt_row.shape)

            # idxs are labels and predictions
            idxs = paddle.stack([pd_row, gt_row], axis=-1)

            updates = paddle.ones([idxs.shape[0]], dtype="int64")

            # make confusion matrix (cols = gt, rows = pred)
            self.conf_matrix = paddle.scatter_nd_add(self.conf_matrix, idxs,
                                                     updates)

    def getStats(self):
        # remove fp from confusion on the ignore classes cols
        conf = self.conf_matrix.clone().astype("float64")
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = paddle.diag(conf, offset=0)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"

    def compute(self, verbose=False) -> dict:
        m_accuracy = self.getacc()
        m_jaccard, class_jaccard = self.getIoU()

        if verbose:
            logger.info("Acc avg {:.3f}".format(float(m_accuracy)))
            logger.info("IoU avg {:.3f}".format(float(m_jaccard)))

            for i, jacc in enumerate(class_jaccard):
                if i not in self.ignore:
                    logger.info(
                        'IoU of class {i:} [{class_str:}] = {jacc:.3f}'.format(
                            i=i,
                            class_str=SemanticKITTIDataset.LABELS[
                                SemanticKITTIDataset.LEARNING_MAP_INV[i]],
                            jacc=float(jacc)))

        return dict(
            mean_acc=m_accuracy, mean_iou=m_jaccard, class_iou=class_jaccard)
