from riglib import experiment
from built_in_tasks.manualcontrolmultitasks import ManualControlMulti
from built_in_tasks.bmimultitasks import BMIControlMulti2DWindow
from riglib.stereo_opengl.window import WindowDispl2D
from features.input_device_features import KeyboardControl, MouseControl, MouseBMI
from features.laser_features import ArduinoGPIO, DigitalWave, LaserTrials
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
    
    def test_exp(self):
        exp = init_exp(BMIControlMulti2DWindow, [MouseBMI])


class TestLaser(unittest.TestCase):
    
    def test_gpio(self):
        gpio = ArduinoGPIO()
    
    def test_digital_wave(self):
        gpio = ArduinoGPIO()
        laser = DigitalWave(gpio, pin=13)
        laser.set_square_wave(5, 5)
        laser.start()
        laser.join()

    def test_laser_trials(self):
        pass


if __name__ == '__main__':
    unittest.main()