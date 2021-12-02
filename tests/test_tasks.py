from built_in_tasks.manualcontrolmultitasks import TrackingTask, rotations, ManualControl
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
    blocks = 1
    trials = 2
    trial_length = 5
    seq = TrackingTask.tracking_target_chain_1D(blocks,trials,trial_length)
    Exp = experiment.make(base_class, feats=feats)
    exp = Exp(seq)
    exp.init()
    return exp

class TestManualControlTasks(unittest.TestCase):
    
    def test_exp(self):
        exp = init_exp(TrackingTask, [MouseControl, Window2D])
        exp.rotation = 'xzy'
        exp.run()

if __name__ == '__main__':
    unittest.main()