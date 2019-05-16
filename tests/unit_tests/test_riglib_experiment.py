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
        sink.sinks = sink.SinkManager()        
        self.exp = MockLogExperiment()

    def test_exp_fsm_output(self):
        """Experiment state sequence follows reference sequence"""
        self.exp.run_sync()
        self.assertEqual(self.exp.event_log,
            [('state1', 'event1to2', 1), ('state2', 'event2to3', 2), ('state3', 'event3to1', 3), 
             ('state1', 'event1to3', 4), ('state3', 'event3to2', 5), ('state2', 'stop', 7)])

    def test_logging(self):
        """LogExperiment should save the sequence of states at cleanup time"""
        mock_db = MockDatabase()
        self.exp.cleanup(mock_db, "id_is_test_file")
        self.assertEqual(str(self.exp.event_log), open("id_is_test_file").readlines()[0].rstrip())
        if os.path.exists("id_is_test_file"):
            os.remove("id_is_test_file")

    def test_cleanup_hdf_when_no_hdf_present(self):
        """experiment.Experiment.cleanup_hdf should run without error when data is *not* being saved to HDF"""
        self.exp.run_sync()
        self.exp.cleanup_hdf()

    def test_add_dtype(self):
        """Experiment should construct a struct of relevant data expected to change on each iteration"""
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
        self.exp.start()
        self.exp.join()
        self.assertEqual(self.exp.event_log,
            [('state1', 'event1to2', 1), ('state2', 'event2to3', 2), ('state3', 'event3to1', 3), 
             ('state1', 'event1to3', 4), ('state3', 'event3to2', 5), ('state2', 'stop', 7)])

        self.assertEqual(self.exp.cycle_count, 7)

    def test_thread_stop(self):
        """Experiment execution in remote thread should terminate when 'end_task' is called"""
        # Experiment should never start if the initial state is None
        exp = experiment.Experiment()
        exp.state = None
        exp.start()
        exp.join()
        self.assertEqual(exp.cycle_count, 0)

        exp = experiment.Experiment()
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

if __name__ == '__main__':
    unittest.main()