'''
Laser delivery features
'''

from riglib.experiment import traits
from riglib.gpio import ArduinoGPIO, DigitalWave
import numpy as np

class CrystaLaser(traits.HasTraits):
    ''' Adds an arduino-controlled crystalaser to self.lasers.'''

    # laser_serial_port = traits.Str(desc="Serial port used to communicate with arduino")
    crystalaser_pin = traits.Int(12, desc="Pin number for laser")

    def __init__(self, *args, **kwargs):
        self.lasers = []
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        laser = ArduinoGPIO('/dev/crystalaser')
        laser.port = self.crystalaser_pin
        laser.name = 'crystalaser'
        self.lasers.append(laser)
        super().init(*args, **kwargs)