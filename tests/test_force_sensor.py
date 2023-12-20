import serial
import pyfirmata, time
import serial.tools.list_ports
import numpy as np
from riglib.gpio import ArduinoGPIO, make

# The following command can be used for finding available ports
# python -m serial.tools.list_ports

import unittest


class ForceSensorTests(unittest.TestCase):

    def test_gpio(self):
        gpio = ArduinoGPIO('/dev/forcesensor', enable_analog=True)
        pin = gpio.board.get_pin('a:0:i')
        pin.enable_reporting()
        t0 = time.time()
        while time.time() - t0 < 50:
            print(pin.read())
            time.sleep(0.1)
        


if __name__ == '__main__':
    unittest.main()