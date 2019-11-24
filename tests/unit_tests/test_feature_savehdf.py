import numpy as np
import unittest
from reqlib import swreq
from requirements import *

from features.hdf_features import SaveHDF
import time
import tables

from riglib import experiment
from riglib import sink
import mocks
import os

class TestExp(experiment.Experiment):
    """ This is an experiment that cycles through a few states and then terminates """
    status = dict(
        wait = dict(start_trial="trial", premature="penalty", stop=None),
        trial = dict(correct="reward", incorrect="penalty", timeout="penalty"),
        reward = dict(post_reward="wait"),
        penalty = dict(post_penalty="wait"),
    )

    def _test_start_trial(self, ts):
        return self.cycle_count == 100

    def _test_correct(self, ts):
        return True

    def _test_post_reward(self, ts):
        return self.cycle_count > 200

    def _test_stop(self, ts):
        return self.cycle_count > 600

    def _cycle(self, *args, **kwargs):
        if self.cycle_count > 250 and self.cycle_count < 300:
            self.task_data["dummy_feat_for_test"] = -1
        else:
            self.task_data["dummy_feat_for_test"] = 0
        super(TestExp, self)._cycle(*args, **kwargs)


class TestSaveHDF(unittest.TestCase):
    def setUp(self):
        sink.sinks = sink.SinkManager()

    def test_save_hdf(self):
        TestFeat = experiment.make(TestExp, feats=[SaveHDF])
        feat = TestFeat()
        feat.add_dtype("dummy_feat_for_test", "f8", (1,))

        # start the feature
        feat.start()

        feat.join()

        mock_db = mocks.MockDatabase()
        feat.cleanup(mock_db, "saveHDF_test_output")
        
        hdf = tables.open_file("saveHDF_test_output.hdf")

        saved_msgs = [x.decode('utf-8') for x in hdf.root.task_msgs[:]["msg"]]
        self.assertEqual(saved_msgs, ['wait', 'trial', 'reward', 'wait', 'None'])
        self.assertEqual(hdf.root.task_msgs[:]["time"].tolist(), [0, 100, 101, 201, 601])

        self.assertTrue(np.all(hdf.root.task[:]["dummy_feat_for_test"][251:300] == -1))
        self.assertTrue(np.all(hdf.root.task[:]["dummy_feat_for_test"][:251] == 0))
        self.assertTrue(np.all(hdf.root.task[:]["dummy_feat_for_test"][300:] == 0))

    def tearDown(self):
        if os.path.exists("saveHDF_test_output.hdf"):
            try:
                os.remove("saveHDF_test_output.hdf")
            except PermissionError:
                pass


if __name__ == '__main__':
    unittest.main()
