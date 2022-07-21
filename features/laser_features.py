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
    '''
    Enable a single QWALOR laser channel
    '''

    # laser_serial_port = traits.Str(desc="Serial port used to communicate with arduino")
    qwalor_channel = traits.Int(1, desc="Laser channel (1-red, 2-blue, 3-green, 4-blue)")
    qwalor_sensor_ach = traits.Int(16, desc="Analog channel (0-index) recording laser power")

    hidden_traits = ['qwalor_sensor_ach']

    def __init__(self, *args, **kwargs):
        self.lasers = []
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):

        # Attempt to open the laser connection, but fail gracefully if it is unavailable
        try:
            laser = QwalorLaserSerial(self.qwalor_channel)
            self.qwalor_laser_status = 'ok'
            
            # Add the laser to the list of available lasers
            laser.port = laser.trigger_pin
            laser.name = 'qwalor_laser'
            self.lasers.append(laser)

        except Exception as e:
            self.qwalor_laser_status = 'Couldn\'t connect to laser modulator, make it is turned on!'
            
        super().init(*args, **kwargs)

    def run(self):
        if not self.qwalor_laser_status == 'ok':
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write(self.qwalor_laser_status)
            self.termination_err.seek(0)
            self.state = None
        super().run()

class MultiQwalorLaser(traits.HasTraits):
    '''
    Enable multiple QWALOR laser channels
    '''

    qwalor_ch1_enable = traits.Bool(False, desc="Laser channel 1-red")
    qwalor_ch2_enable = traits.Bool(False, desc="Laser channel 2-blue")
    qwalor_ch3_enable = traits.Bool(False, desc="Laser channel 3-green")
    qwalor_ch4_enable = traits.Bool(False, desc="Laser channel 4-blue")
    qwalor_ch1_sensor_ach = traits.Int(15, desc="Analog channel (0-index) recording laser ch1 power")
    qwalor_ch2_sensor_ach = traits.Int(16, desc="Analog channel (0-index) recording laser ch2 power")
    qwalor_ch3_sensor_ach = traits.Int(17, desc="Analog channel (0-index) recording laser ch3 power")
    qwalor_ch4_sensor_ach = traits.Int(18, desc="Analog channel (0-index) recording laser ch4 power")

    hidden_traits = ['qwalor_ch1_sensor_ach', 'qwalor_ch2_sensor_ach', 'qwalor_ch3_sensor_ach', 'qwalor_ch4_sensor_ach']

    def __init__(self, *args, **kwargs):
        self.lasers = []
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):

        # Attempt to open the laser connections, but fail gracefully if it is unavailable
        try:
            if self.qwalor_ch1_enable:
                laser = QwalorLaserSerial(1)
                laser.port = laser.trigger_pin
                laser.name = 'qwalor_laser_ch1'
                self.lasers.append(laser)
            
            if self.qwalor_ch2_enable:
                laser = QwalorLaserSerial(2)
                laser.port = laser.trigger_pin
                laser.name = 'qwalor_laser_ch2'
                self.lasers.append(laser)

            if self.qwalor_ch3_enable:
                laser = QwalorLaserSerial(3)
                laser.port = laser.trigger_pin
                laser.name = 'qwalor_laser_ch3'
                self.lasers.append(laser)

            if self.qwalor_ch4_enable:
                laser = QwalorLaserSerial(4)
                laser.port = laser.trigger_pin
                laser.name = 'qwalor_laser_ch4'
                self.lasers.append(laser)

            self.qwalor_laser_status = 'ok'
            
        except Exception as e:
            self.qwalor_laser_status = 'Couldn\'t connect to laser modulator, make it is turned on!'
            
        super().init(*args, **kwargs)

    def run(self):
        if not self.qwalor_laser_status == 'ok':
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write(self.qwalor_laser_status)
            self.termination_err.seek(0)
            self.state = None
        super().run()