from riglib import experiment
from built_in_tasks.manualcontrolmultitasks import ManualControl
from riglib.stereo_opengl.window import WindowDispl2D
from features.peripheral_device_features import KeyboardControl, MouseControl
from features.laser_features import LaserTrials
import features.sync_features as sync_features
import numpy as np

import unittest

def init_exp(base_class, feats):
    blocks = 1
    targets = 3
    seq = ManualControl.centerout_2D_discrete(blocks, targets)
    Exp = experiment.make(base_class, feats=feats)
    exp = Exp(seq)
    exp.init()
    return exp

class TestKeyboardControl(unittest.TestCase):

    def setUp(self):
        pass
    
    @unittest.skip("msg")
    def test_exp(self):
        exp = init_exp(ManualControl, [KeyboardControl, WindowDispl2D])
        exp.run()

class TestMouseControl(unittest.TestCase):

    @unittest.skip("msg")
    def test_exp(self):
        exp = init_exp(ManualControl, [MouseControl, WindowDispl2D])
        exp.run()

class TestLaser(unittest.TestCase):
    
    def test_digital_wave(self):
        from riglib.gpio import TestGPIO, DigitalWave
        gpio = TestGPIO()
        laser1 = DigitalWave(gpio, pin=1)
        laser1.set_pulse(1, 0)
        self.assertCountEqual(laser1.edges, [0, 1])
        laser1.set_pulse(1, 1)
        self.assertCountEqual(laser1.edges, [0, 1])
        laser1.set_square_wave(1, 5)
        self.assertCountEqual(laser1.edges, np.linspace(0, 5.0, 11))
        laser1.start()
        laser1.join()
        self.assertEqual(sum(gpio.value[1,:]), 6) # 1 Hz over 5 seconds has 6 positive edges including the last one at 5 s

    @unittest.skip("Need arduino connected for this to pass")
    def test_arduino(self):
        from riglib.gpio import ArduinoGPIO, DigitalWave
        gpio = ArduinoGPIO()
        laser = DigitalWave(gpio, pin=10)
        laser.set_square_wave(5, 10)
        laser.start()
        laser.join()
        laser = DigitalWave(gpio, pin=10)
        laser.set_edges([0], False)
        laser.start()
        laser.join()

    def test_laser_trials(self):
        pass

class TestSync(unittest.TestCase):

    def test_dictionary(self):
        default_dict = sync_features.sync_protocol
        self.assertEqual(default_dict['TARGET_ON'] + 4, sync_features.encode_event(default_dict, 'TARGET_ON', 4))
        for k in default_dict.keys():
            event_data = 0
            encode = sync_features.encode_event(default_dict, k, event_data)
            decode = sync_features.decode_event(default_dict, encode)
            self.assertEqual(decode[0], k)    
            self.assertEqual(decode[1], event_data)


if __name__ == '__main__':
    unittest.main()