from riglib import experiment
from built_in_tasks.manualcontrolmultitasks import ManualControl
from built_in_tasks.othertasks import LaserConditions
from riglib.stereo_opengl.window import WindowDispl2D
from features.peripheral_device_features import KeyboardControl, MouseControl
import features.sync_features as sync_features
from features.laser_features import CrystaLaser
from features.video_recording_features import E3Video
from riglib.e3vision import E3VisionInterface
import numpy as np
import time

import unittest


def init_exp(base_class, feats, seq=None, **kwargs):
    blocks = 2
    trials = 2
    trial_length = 5
    frequencies = np.array([.5])
    Exp = experiment.make(base_class, feats=feats)
    if seq is not None:
        exp = Exp(seq, **kwargs)
    else:
        exp = Exp(**kwargs)
    exp.init()
    return exp

class TestKeyboardControl(unittest.TestCase):

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
        laser1 = DigitalWave(gpio, mask=2)
        laser1.set_pulse(1, 0)
        self.assertCountEqual(laser1.edges, [0, 1])
        laser1.set_pulse(1, 1)
        self.assertCountEqual(laser1.edges, [0, 1])
        laser1.set_square_wave(1, 5)
        self.assertCountEqual(laser1.edges, np.linspace(0, 5.0, 11))
        laser1.run() # run on the same thread so we can make changes to the test gpio state
        print(gpio.value[1,:])
        self.assertEqual(sum(gpio.value[1,:]), 6) # 1 Hz over 5 seconds has 6 positive edges including the last one at 5 s

        # Test duty cycle
        laser2 = DigitalWave(gpio, mask=2)
        laser2.set_square_wave(1, 5, duty_cycle=0.2)
        expected = [0. , 0.2, 1. , 1.2, 2. , 2.2, 3. , 3.2, 4. , 4.2, 5. ]
        self.assertCountEqual(laser2.edges, expected)

        # Test phase delay
        laser2 = DigitalWave(gpio, mask=2)
        laser2.set_square_wave(1, 5, phase_delay=0.2)
        self.assertCountEqual(laser2.edges, np.linspace(0.2, 5.2, 11))

    def test_convert_masked_data_to_pins(self):
        from riglib.gpio import convert_masked_data_to_pins
        pins, values = convert_masked_data_to_pins(1<<12, 1<<12)
        self.assertEqual(pins, [12])
        self.assertTrue(values)

    @unittest.skip("Need arduino connected for this to pass")
    def test_arduino(self):
        from riglib.gpio import ArduinoGPIO, DigitalWave
        gpio = ArduinoGPIO('/dev/crystalaser')
        ch = 12
        laser = DigitalWave(gpio, mask=1<<ch)
        laser.set_square_wave(5, 10)
        laser.start()
        laser.join()
        laser = DigitalWave(gpio, mask=1<<ch)
        laser.set_edges([0], False)
        laser.start()
        laser.join()

    @unittest.skip("Need arduino connected for this to pass")
    def test_laser_conditions(self):
        Exp = experiment.make(LaserConditions, [CrystaLaser])
        exp = Exp(LaserConditions.single_laser_pulse(nreps=2))
        exp.init()
        exp.run()

    def test_dio_pulse_width(self):
        from riglib.gpio import ArduinoGPIO, DigitalWave
        import time
        gpio = ArduinoGPIO('/dev/crystalaser')

        pulse_widths = [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001]
        reps = 20
        durations = np.zeros((len(pulse_widths),reps))
        for width_idx, width in enumerate(pulse_widths):
            for n in range(reps):
                t0 = time.perf_counter()
                gpio.write_many(1<<12, 1<<12)
                while (time.perf_counter() - t0 < width):
                    pass
                durations[width_idx, n] = time.perf_counter() - t0
                gpio.write_many(1<<12, 0)
                time.sleep(0.1)

        print(np.mean(durations, axis=1))
        print(np.std(durations, axis=1))

        time.sleep(1)

        seq = LaserConditions.single_laser_pulse(nreps=reps, duration=pulse_widths, uniformsampling=False, ascending=True)
        for idx, powers, edges in seq:
            edges = edges[0]
            wave = DigitalWave(gpio, mask=1<<12)
            wave.set_edges(edges, True)
            wave.start()
            wave.join()
            time.sleep(0.1)

class TestSync(unittest.TestCase):

    def test_dictionary(self):
        default_dict = sync_features.rig1_sync_params['event_sync_dict']
        self.assertEqual(default_dict['TARGET_ON'] + 4, sync_features.encode_event(default_dict, 'TARGET_ON', 4))
        for k in default_dict.keys():
            event_data = 0
            encode = sync_features.encode_event(default_dict, k, event_data)
            decode = sync_features.decode_event(default_dict, encode)
            self.assertEqual(decode[0], k)    
            self.assertEqual(decode[1], event_data)

class TestE3Video(unittest.TestCase):

    # def test_interface(self):
    #     e3v = E3VisionInterface()
    #     e3v.update_camera_status()
    #     e3v.start_rec()
    #     time.sleep(5)
    #     e3v.stop_rec()

    def test_feature(self):
        exp = init_exp(experiment.Experiment, [E3Video], saveid=0)
        exp.run()
        time.sleep(1)
        exp.state = None

if __name__ == '__main__':
    unittest.main()