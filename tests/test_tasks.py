from built_in_tasks.manualcontrolmultitasks import ManualControl
from riglib.stereo_opengl.window import WindowDispl2D
from built_in_tasks.othertasks import LaserConditions
from riglib import experiment
import cProfile
import pstats

import unittest
import numpy as np

def init_exp(base_class, feats):
    blocks = 1
    targets = 3
    seq = ManualControl.centerout_2D(blocks, targets)
    Exp = experiment.make(base_class, feats=feats)
    exp = Exp(seq)
    exp.init()
    return exp

class TestManualControlTasks(unittest.TestCase):
    
    def test_exp(self):
        pr = cProfile.Profile()
        pr.enable()
        exp = init_exp(ManualControl, [WindowDispl2D])
        exp.run()
        pr.disable()
        with open('profile.csv', 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('time')
            ps.print_stats()

class TestOtherTasks(unittest.TestCase):

    def test_gen(self):
        powers, edges = LaserConditions.pulse(10, [0.005], [1])
        self.assertCountEqual(powers, np.ones(10))
        for edge in edges:
            self.assertCountEqual(edge, [0, 0.005])

    def test_exp(self):
        exp = init_exp(LaserConditions, [])
        exp.run()

if __name__ == '__main__':
    unittest.main()