'''
Laser delivery features
'''

from riglib.experiment import traits
from riglib.gpio import ArduinoGPIO, DigitalWave
import numpy as np
from riglib.qwalor_laser import QwalorLaserSerial

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
        laser.set_power = lambda x: None
        self.lasers.append(laser)
        super().init(*args, **kwargs)

class QwalorLaser(traits.HasTraits):

    # laser_serial_port = traits.Str(desc="Serial port used to communicate with arduino")
    qwalor_trigger_pin = traits.Int(12, desc="Pin number for laser trigger")
    qwalor_channel = traits.Int(1, desc="Laser channel (1-red, 2-blue, 3-green, 4-blue)")

    def __init__(self, *args, **kwargs):
        self.lasers = []
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        laser = QwalorLaserSerial(self.qwalor_channel, '/dev/ttyACM1', self.qwalor_trigger_pin)
        laser.port = self.qwalor_trigger_pin
        laser.name = 'qwalor_laser'
        self.lasers.append(laser)
        super().init(*args, **kwargs)