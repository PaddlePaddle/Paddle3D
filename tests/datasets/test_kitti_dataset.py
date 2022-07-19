import unittest

import numpy as np
import paddle

import paddle3d


class KittiMonoDatasetTestCase(unittest.TestCase):
    """
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #prepare dataset to temp dir
        self.kitti_train = paddle3d.datasets.KittiMonoDataset(
            dataset_root='../../datasets/KITTI/')
        self.kitti_trainval = paddle3d.datasets.KittiMonoDataset(
            dataset_root='../../datasets/KITTI/', mode='trainval')
        self.kitti_val = paddle3d.datasets.KittiMonoDataset(
            dataset_root='../../datasets/KITTI/', mode='val')
        self.kitti_test = paddle3d.datasets.KittiMonoDataset(
            dataset_root='../../datasets/KITTI/', mode='test')

    def test_size(self):
        """
        """
        self.assertEqual(len(self.kitti_train), 3712)
        self.assertEqual(len(self.kitti_trainval), 7480)
        self.assertEqual(len(self.kitti_val), 3769)
        self.assertEqual(len(self.kitti_test), 7517)

    def test_evaluation(self):
        """
        """
        samples = [s for s in self.kitti_train]
        # add confidences
        for s in samples:
            num_boxes = s.bboxes_2d.shape[0]
            s.confidences = np.ones([num_boxes])

        metric_obj = self.kitti_train.metric
        metric_obj.update(samples)
        print(metric_obj.compute())

    def test_batching(self):
        loader = paddle.io.DataLoader(
            self.kitti_train,
            batch_size=4,
            collate_fn=self.kitti_train.collate_fn)
        for _ in loader:
            ...


class KittiPCDatasetTestCase(unittest.TestCase):
    """
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #prepare dataset to temp dir
        self.kitti_train = paddle3d.datasets.KittiPCDataset(
            dataset_root='../../datasets/KITTI/')
        self.kitti_trainval = paddle3d.datasets.KittiPCDataset(
            dataset_root='../../datasets/KITTI/', mode='trainval')
        self.kitti_val = paddle3d.datasets.KittiPCDataset(
            dataset_root='../../datasets/KITTI/', mode='val')
        self.kitti_test = paddle3d.datasets.KittiPCDataset(
            dataset_root='../../datasets/KITTI/', mode='test')

    def test_size(self):
        """
        """
        self.assertEqual(len(self.kitti_train), 3712)
        self.assertEqual(len(self.kitti_trainval), 7480)
        self.assertEqual(len(self.kitti_val), 3769)
        self.assertEqual(len(self.kitti_test), 7517)

    def test_evaluation(self):
        """
        """
        samples = [s for s in self.kitti_train]
        # add confidences
        for s in samples:
            num_boxes = s.bboxes_3d.shape[0]
            s.confidences = np.ones([num_boxes])

        metric_obj = self.kitti_train.metric
        metric_obj.update(samples)
        print(metric_obj.compute())

    def test_batching(self):
        loader = paddle.io.DataLoader(
            self.kitti_train,
            batch_size=4,
            collate_fn=self.kitti_train.collate_fn)
        for _ in loader:
            ...


if __name__ == "__main__":
    unittest.main()
