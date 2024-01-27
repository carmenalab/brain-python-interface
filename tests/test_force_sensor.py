import serial
import pyfirmata, time
import serial.tools.list_ports
import numpy as np
from riglib.gpio import ArduinoGPIO
from features.peripheral_device_features import ForceSensorControl

# The following command can be used for finding available ports
# python -m serial.tools.list_ports

import unittest


class ForceSensorTests(unittest.TestCase):

    @unittest.skip("")
    def test_gpio(self):
        gpio = ArduinoGPIO('/dev/forcesensor', enable_analog=True)
        t0 = time.time()
        while time.time() - t0 < 5:
            print(gpio.analog_read(0))
            time.sleep(0.1)
        
    def test_source(self):
        f = ForceSensorControl()
        f.init()
        t0 = time.time()
        while time.time() - t0 < 5:
            print(f.force_sensor.get())
            time.sleep(0.1)


if __name__ == '__main__':
    unittest.main()