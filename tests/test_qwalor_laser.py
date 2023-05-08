import serial
import pyfirmata, time
import serial.tools.list_ports
import numpy as np
from riglib import qwalor_laser

# The following command can be used for finding available ports
# python -m serial.tools.list_ports

import unittest

# laser setting
channel = 2
mode = 'off'
freq = 0

# pulse_width_list = [0.00001, 0.00002, 0.00003, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
pulse_width_list = [1]
power_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

class LaserTests(unittest.TestCase):

    def test_config(self):
        packet = qwalor_laser.get_config_packet(1, 0, 0, debug=True)
        self.assertEqual(np.count_nonzero(packet), 0)

        packet = qwalor_laser.get_config_packet(2, 0.5137, 0.5, mode='SW', debug=True)
        self.assertEqual(np.count_nonzero(packet), 4)
        self.assertEqual(packet[1], 1)
        self.assertEqual(packet[2], 0xff)
        self.assertEqual(packet[3], 0x7f)

    @unittest.skip("only run if needed")
    def test_run(self):

        laser = qwalor_laser.QwalorLaserSerial(channel)
        laser.set_mode(mode)
        laser.set_freq(freq)

        for power in power_list:
            laser.set_power(power)
            print(power)
            for width in pulse_width_list:
                t0 = time.perf_counter()
                laser.on()
                while (time.perf_counter() - t0 < width):
                    pass
                laser.off()
                time.sleep(0.5)

    def test_dual_channel(self):
        n_trials = 10
        iti = 0.1
        width = 0.1

        laser1 = qwalor_laser.QwalorLaserSerial(2)
        laser1.port = laser1.trigger_pin
        laser1.name = 'qwalor_laser_ch2'
        print("laser 1 configured")

        laser2 = qwalor_laser.QwalorLaserSerial(4)
        laser2.port = laser2.trigger_pin
        laser2.name = 'qwalor_laser_ch4'
        print("laser 2 configured")

        for n in range(n_trials):
            t0 = time.perf_counter()
            laser1.set_mode(mode)
            laser1.set_freq(freq)
            laser1.set_power(1)
            laser1.on()
            while (time.perf_counter() - t0 < width):
                pass
            laser1.off()
            t1 = time.perf_counter()
            while (time.perf_counter() - t1 < iti):
                pass
        
            t0 = time.perf_counter()
            laser2.set_mode(mode)
            laser2.set_freq(freq)
            laser2.set_power(1)
            laser2.on()
            while (time.perf_counter() - t0 < width):
                pass
            laser2.off()
            t1 = time.perf_counter()
            while (time.perf_counter() - t1 < iti):
                pass


    @unittest.skip("only run if needed")
    def test_speed(self):

        laser = qwalor_laser.QwalorLaserSerial(channel)
        laser.set_mode(mode)
        laser.set_freq(freq)

        n_trials = 1000
        iti = 0.001
        width = 0.001

        t_begin = time.perf_counter()

        for n in range(n_trials):
            laser.set_power(1)
            t0 = time.perf_counter()
            laser.on()
            while (time.perf_counter() - t0 < width):
                pass
            laser.off()
            t1 = time.perf_counter()
            while (time.perf_counter() - t1 < iti):
                pass

        t_end = time.perf_counter()
        print(f"laser took {t_end-t_begin:.2f} s to run {n_trials} trials of {width} width and {iti} iti")
        time.sleep(0.5)
        laser.off()


if __name__ == '__main__':
    unittest.main()