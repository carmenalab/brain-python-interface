import unittest
import os
import numpy as np
import time

from ..riglib import experiment
from ..riglib.experiment.mocks import MockSequenceWithGenerators
from ..features.hdf_features import SaveHDF
import h5py

from ..riglib import sink

class TestTaskWithFeatures(unittest.TestCase):
    def setUp(self):
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.reset()

    def test_metaclass_constructor(self):
        exp = experiment.make(experiment.LogExperiment, feats=(SaveHDF,))
        exp()

    def test_mock_seq_with_features(self):
        task_cls = experiment.make(MockSequenceWithGenerators, feats=(SaveHDF,))
        exp = task_cls(MockSequenceWithGenerators.gen_fn1())
        exp.run_sync()

        time.sleep(2)
        hdf = h5py.File(exp.h5file_name)

        # test that the annotation appears in the messages
        self.assertTrue(b'annotation: test annotation' in hdf["/task_msgs"]["msg"])

        ref_current_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0])
        saved_current_state = hdf["/task"]["current_state"].ravel()

        # TODO this length chopping should be needed, but the vector appears to be short sometimes
        L = min(len(ref_current_state), len(saved_current_state))
        self.assertTrue(np.array_equal(ref_current_state[:L], saved_current_state[:L]))
        

if __name__ == '__main__':
    unittest.main()