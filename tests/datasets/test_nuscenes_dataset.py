import unittest

import numpy as np
import paddle

import paddle3d


class NuscenesPCDatasetTestCase(unittest.TestCase):
    """
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #prepare dataset to temp dir
        self.nuscenes_minitrain = paddle3d.datasets.NuscenesPCDataset(
            dataset_root='../../datasets/nuscenes/', mode='mini_train')
        self.nuscenes_minival = paddle3d.datasets.NuscenesPCDataset(
            dataset_root='../../datasets/nuscenes/', mode='mini_val')

    def test_size(self):
        """
        """
        self.assertEqual(len(self.nuscenes_minitrain), 323)
        self.assertEqual(len(self.nuscenes_minival), 81)

    def test_evaluation(self):
        """
        """
        samples = [s for s in self.nuscenes_minitrain]
        # add confidences
        for s in samples:
            num_boxes = s.bboxes_3d.shape[0]
            s.confidences = np.ones([num_boxes])

        metric_obj = self.nuscenes_minitrain.metric
        metric_obj.update(samples)
        print(metric_obj.compute())

    def test_batching(self):
        loader = paddle.io.DataLoader(
            self.nuscenes_minitrain,
            batch_size=4,
            collate_fn=self.nuscenes_minitrain.collate_fn)
        for _ in loader:
            ...


if __name__ == "__main__":
    unittest.main()
