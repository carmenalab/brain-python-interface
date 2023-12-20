import serial
import pyfirmata, time
import serial.tools.list_ports
import numpy as np
from riglib.gpio import ArduinoGPIO, make
from riglib.source import DataSource

# The following command can be used for finding available ports
# python -m serial.tools.list_ports

import unittest


class ForceSensorTests(unittest.TestCase):

    def test_gpio(self):
        gpio = ArduinoGPIO('/dev/forcesensor', enable_analog=True)
        t0 = time.time()
        while time.time() - t0 < 5:
            print(gpio.analog_read(0))
            time.sleep(0.1)
        
    def test_source(self):
        ds = DataSource(make())
        ds.start()
        time.sleep(STREAMING_DURATION)
        data = ds.get_new(channels)
        ds.stop()
        data = np.array(data)

        n_samples = int(Broadband.update_freq * STREAMING_DURATION / 728) * 728 # closest multiple of 728 (floor)

        self.assertEqual(data.shape[0], len(channels))


if __name__ == '__main__':
    unittest.main()