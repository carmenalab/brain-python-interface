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
        sink.sinks = sink.SinkManager() # reset the sink manager

    def test_n_successful_targets(self):
        TestFeat = experiment.make(TargetCaptureVFB2DWindow, feats=[SaveHDF])

        n_targets = 1
        seq = TargetCaptureVFB2DWindow.centerout_2D_discrete(nblocks=1, ntargets=n_targets) 
        feat = TestFeat(seq, window_size=(480, 240))

        feat.run_sync()

        time.sleep(1) # small delay to allow HDF5 file to be written
        hdf = tables.open_file(feat.h5file.name)

        self.assertEqual(n_targets, np.sum(hdf.root.task_msgs[:]['msg'] == b'reward'))

if __name__ == '__main__':
    unittest.main()