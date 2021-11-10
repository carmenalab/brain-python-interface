import unittest
import os
import numpy as np
import time

from riglib.experiment.mocks import *
from riglib import experiment
from riglib import sink

from mocks import MockDatabase


class TestLogExperiment(unittest.TestCase):
    def setUp(self):
        sink_manager = sink.SinkManager.get_instance()
        sink_manager.reset()
        self.exp = MockLogExperiment(verbose=False)

    def test_exp_fsm_output(self):
        """Experiment state sequence follows reference sequence"""
        self.exp = MockLogExperiment(verbose=False)
        self.exp.run_sync()
        self.assertEqual(self.exp.event_log,
            [('state1', 'event1to2', 1), ('state2', 'event2to3', 2), ('state3', 'event3to1', 3),
             ('state1', 'event1to3', 4), ('state3', 'event3to2', 5), ('state2', 'stop', 7)])

    def test_logging(self):
        """LogExperiment should save the sequence of states at cleanup time"""
        mock_db = MockDatabase()
        exp = MockLogExperiment(verbose=False)
        exp.run_sync()
        exp.cleanup(mock_db, "id_is_test_file")

        summary_data = str(exp.log_summary(exp.event_log))

        self.assertEqual(summary_data, open("id_is_test_file").readlines()[0].rstrip())
        if os.path.exists("id_is_test_file"):
            os.remove("id_is_test_file")

    def test_cleanup_hdf_when_no_hdf_present(self):
        """experiment.Experiment.cleanup_hdf should run without error when data is *not* being saved to HDF"""
        self.exp = MockLogExperiment(verbose=False)
        self.exp.run_sync()
        self.exp.cleanup_hdf()

    def test_add_dtype(self):
        """Experiment should construct a struct of relevant data expected to change on each iteration"""
        self.exp = MockLogExperiment(verbose=False)
        self.exp.add_dtype("field1", "float", (1,))

        with self.assertRaises(Exception) as context:
            self.exp.add_dtype("field1", "float", (1,))

        self.assertTrue('Duplicate add_dtype functionc call for task data field' in str(context.exception))

        self.exp.add_dtype("field2", "int", (1,))

        self.exp.init()
        import sys
        if sys.platform == "win32":
            ref_dtype = np.dtype([('field1', '<f8', (1,)), ('field2', '<i4', (1,))])
        else:
            ref_dtype = np.dtype([('field1', '<f8', (1,)), ('field2', '<i8', (1,))])
        self.assertEqual(self.exp.dtype, ref_dtype)

    def test_thread_start(self):
        """Experiment FSM execution should run in a new thread and return when state sequence is complete."""
        self.exp = MockLogExperiment(verbose=False)
        self.exp.start()
        self.exp.join()
        self.assertEqual(self.exp.event_log,
            [('state1', 'event1to2', 1), ('state2', 'event2to3', 2), ('state3', 'event3to1', 3),
             ('state1', 'event1to3', 4), ('state3', 'event3to2', 5), ('state2', 'stop', 7)])

        self.assertEqual(self.exp.cycle_count, 7)

    def test_thread_stop(self):
        """Experiment execution in remote thread should terminate when 'end_task' is called"""
        # Experiment should never start if the initial state is None
        exp = experiment.Experiment(verbose=False)
        exp.state = None
        exp.start()
        exp.join()
        self.assertEqual(exp.cycle_count, 0)

        exp = experiment.Experiment(verbose=False)
        exp.state = "wait"
        exp.start()
        time.sleep(1)
        exp.end_task()

        # number of cycles should be about 60. Wide margin so that test always passes
        # (not exact due to threading and imprecise timing)
        margin = 20
        self.assertTrue(exp.cycle_count > exp.fps - margin)
        self.assertTrue(exp.cycle_count < exp.fps + margin)


class TestSequence(unittest.TestCase):
    def setUp(self):
        self.targets = [1, 2, 3, 4, 5]
        self.exp = MockSequence(self.targets)

    def test_target_sequence(self):
        self.exp.run_sync()
        self.assertEqual(self.targets, self.exp.target_history)


class TestTaskWithFeatures(unittest.TestCase):
    def test_metaclass_constructor(self):
        from features.hdf_features import SaveHDF
        exp = experiment.make(experiment.LogExperiment, feats=(SaveHDF,))
        exp()

    def test_mock_seq_with_features(self):
        from riglib.experiment.mocks import MockSequenceWithGenerators
        from features.hdf_features import SaveHDF
        import h5py

        task_cls = experiment.make(MockSequenceWithGenerators, feats=(SaveHDF,))
        exp = task_cls(MockSequenceWithGenerators.gen_fn1())
        exp.run_sync()

        # optional delay if the OS is not ready to let you open the HDF file
        # created just now
        open_count = 0
        while open_count < 3:
            try:
                hdf = h5py.File(exp.h5file.name)
                break
            except OSError:
                open_count += 1
                if open_count >= 3:
                    raise Exception("Unable to open HDF file!")
                time.sleep(1)

        ref_current_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0])
        self.assertTrue(np.array_equal(hdf["/task"]["current_state"].ravel(), ref_current_state))

if __name__ == '__main__':
    unittest.main()