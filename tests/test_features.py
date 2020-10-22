from riglib import experiment
from built_in_tasks.manualcontrolmultitasks import ManualControlMulti
from built_in_tasks.bmimultitasks import BMIControlMulti2DWindow
from riglib.stereo_opengl.window import WindowDispl2D
from features.peripheral_device_features import KeyboardControl, MouseControl, MouseBMI
from features.arduino_features import FakeGPIOFeature, ArduinoGPIOFeature
from features.laser_features import DigitalWave, LaserTrials
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
    
    @unittest.skip("msg")
    def test_exp(self):
        exp = init_exp(ManualControlMulti, [KeyboardControl, WindowDispl2D])
        exp.run()

class TestMouseControl(unittest.TestCase):

    @unittest.skip("msg")
    def test_exp(self):
        exp = init_exp(ManualControlMulti, [MouseControl, WindowDispl2D])
        exp.run()

class TestMouseBMI(unittest.TestCase):
    
    @unittest.skip("msg")
    def test_exp(self):
        exp = init_exp(BMIControlMulti2DWindow, [MouseBMI])


class TestLaser(unittest.TestCase):
    
    def test_digital_wave(self):
        exp = FakeGPIOFeature()
        laser1 = DigitalWave(exp.gpio, pin=13)
        laser1.set_square_wave(1, 5)
        self.assertCountEqual(laser1.edges, np.linspace(0, 5.0, 11))
        laser1.start()
        laser1.join()

    def test_arduino(self):
        exp = ArduinoGPIOFeature()
        laser = DigitalWave(exp.gpio, pin=10)
        laser.set_square_wave(5, 10)
        laser.start()
        laser.join()
        laser = DigitalWave(exp.gpio, pin=10)
        laser.set_edges([0], False)
        laser.start()
        laser.join()

    def test_laser_trials(self):
        pass


if __name__ == '__main__':
    unittest.main()