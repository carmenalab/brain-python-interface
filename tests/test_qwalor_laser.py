import serial
import pyfirmata, time
import serial.tools.list_ports
import numpy as np
from riglib import qwalor_laser

# The following command can be used for finding available ports
# python -m serial.tools.list_ports

import unittest

# laser setting
channel = 4
mode = 'off'
freq = 0
b_rate = 115200
port_laser = '/dev/qwalorlaser'

# pulse_width_list = [0.00001, 0.00002, 0.00003, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
pulse_width_list = [5]
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

    def test_run(self):

        laser = qwalor_laser.QwalorLaserSerial(channel, laser_port=port_laser, laser_baud_rate=b_rate)
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

if __name__ == '__main__':
    unittest.main()