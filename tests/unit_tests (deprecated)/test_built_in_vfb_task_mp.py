#!/usr/bin/env python
import os
import unittest
import time
import tables
import numpy as np

from reqlib import swreq
from requirements import *
from features.hdf_features import SaveHDF
from riglib import experiment
from riglib import sink

from built_in_tasks.passivetasks import TargetCaptureVFB2DWindow

class TestVisualFeedback(unittest.TestCase):
    def setUp(self):
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.reset()

        n_targets = 1
        nblocks = 1
        seq = TargetCaptureVFB2DWindow.centerout_2D_discrete(nblocks=nblocks, ntargets=n_targets) 

        params = dict(window_size=(480, 240))

        task_class = experiment.make(TargetCaptureVFB2DWindow, feats=[SaveHDF])

        task_wrapper = experiment.task_wrapper.TaskWrapper(
            subj=None, params=params,
            target_class=task_class,
            seq=seq, log_filename='tasktrack_log')

        task_proxy, data_proxy = task_wrapper.start()
        self.task_proxy = task_proxy
        self.task_wrapper = task_wrapper
        self.n_trials = n_targets * nblocks

    def test_end_task_early_through_proxy(self):
        time.sleep(1) # delay to allow file to be created
        h5file_name = self.task_proxy.h5file_name

        self.task_wrapper.end_task()

        print("sleeping to let the task finish")
        time.sleep(2)

        self.task_wrapper.join()

        hdf = tables.open_file(h5file_name)

        msgs = hdf.root.task_msgs[:]['msg']
        n_complete_trials = np.sum(msgs == b'reward')
        self.assertTrue(n_complete_trials < self.n_trials)        


if __name__ == '__main__':
    unittest.main()
