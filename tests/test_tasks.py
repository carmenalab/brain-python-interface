from built_in_tasks.manualcontrolmultitasks import TrackingTask, rotations, ManualControl
from built_in_tasks.othertasks import Conditions, LaserConditions
from built_in_tasks.target_capture_task import ScreenTargetCapture
from features.hdf_features import SaveHDF
from riglib.stereo_opengl.window import WindowDispl2D
from built_in_tasks.othertasks import LaserConditions
from riglib import experiment
from features.peripheral_device_features import MouseControl
import cProfile
import pstats
from riglib.stereo_opengl.window import Window, Window2D 
import unittest
import numpy as np



def init_exp(base_class, feats):
    blocks = 2
    trials = 2
    trial_length = 5
    frequencies = np.array([.5])
    seq = TrackingTask.tracking_target_training(blocks,trials,trial_length,frequencies)
    Exp = experiment.make(base_class, feats=feats)
    exp = Exp(seq)
    exp.init()
    return exp

class TestManualControlTasks(unittest.TestCase):
    
    @unittest.skip("")
    def test_exp(self):
        exp = init_exp(TrackingTask, [MouseControl, Window2D])
        exp.rotation = 'xzy'
        exp.run()

class TestSeqGenerators(unittest.TestCase):

    def test_gen_ascending(self):
        seq = Conditions.gen_conditions(3, [1, 2], ascend=True)
        self.assertSequenceEqual(seq[0], [0, 0, 0, 1, 1, 1])

    def test_gen_out_2D(self):
        seq = ScreenTargetCapture.out_2D(nblocks=1, )
        seq = list(seq)
        idx = np.array([s[0][0] for s in seq])
        loc = np.array([s[1][0] for s in seq])
        print(idx)
        print(loc)
        self.assertCountEqual(idx, [1, 2, 3, 4, 5, 6, 7, 8])

        # Target 1 should be 12 o'clock
        self.assertAlmostEqual(loc[idx == 1, 0][0], 0)
        self.assertAlmostEqual(loc[idx == 1, 2][0], 10)

        # Target 3 should be 3 o'clock
        self.assertAlmostEqual(loc[idx == 3, 0][0], 10)
        self.assertAlmostEqual(loc[idx == 3, 2][0], 0)

if __name__ == '__main__':
    unittest.main()