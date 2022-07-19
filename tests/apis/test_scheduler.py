import unittest

import paddle3d


class SchedulerTestCase(unittest.TestCase):
    """
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = paddle3d.apis.Scheduler(
            save_interval=10, log_interval=5, do_eval=True)

    def test_status(self):
        for i in range(1, 21):
            status = self.scheduler.step()
            if i % 5 == 0:
                self.assertEqual(status.do_log, True)
            else:
                self.assertEqual(status.do_log, False)

            if i % 10 == 0:
                self.assertEqual(status.save_checkpoint, True)
                self.assertEqual(status.do_eval, True)
            else:
                self.assertEqual(status.save_checkpoint, False)
                self.assertEqual(status.do_eval, False)


if __name__ == "__main__":
    unittest.main()
