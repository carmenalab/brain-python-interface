from riglib import experiment
from built_in_tasks.manualcontrolmultitasks import ManualControlMulti
from riglib.stereo_opengl.window import WindowDispl2D
from features.input_device_features import KeyboardControl, MouseControl
import numpy as np

import unittest

def init_exp(base_class, feats):
    blocks = 1
    targets = 3
    seq = ManualControlMulti.centerout_2D_discrete(blocks, targets)
    Exp = experiment.make(base_class, feats=feats)
    exp = Exp(seq)
    exp.init()
    return exp

class TestKeyboardControl(unittest.TestCase):

    def setUp(self):
        pass

    def test_exp(self):
        exp = init_exp(ManualControlMulti, [KeyboardControl, WindowDispl2D])
        exp.run()

class TestMouseControl(unittest.TestCase):

    def test_exp(self):
        exp = init_exp(ManualControlMulti, [MouseControl, WindowDispl2D])
        exp.run()

if __name__ == '__main__':
    unittest.main()


