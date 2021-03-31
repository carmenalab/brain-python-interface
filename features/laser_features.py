'''
Laser delivery features
'''

from riglib.experiment import traits
from riglib.gpio import ArduinoGPIO, DigitalWave
import numpy as np

class LaserTrials(traits.HasTraits):
    ''' Activate a laser at the start of each trial. Must have a GPIO feature enabled'''

    # laser_serial_port = traits.Str(desc="Serial port used to communicate with arduino")
    laser_gpio_pin = traits.Int(12, desc="Pin number for laser")
    laser_wave = traits.OptionsList(("Constant", "Pulse", "Square wave"), desc="Laser wave type")
    laser_first_edge = traits.Bool(True, desc="Laser first edge")
    laser_duration = traits.Float(desc="Laser duration (seconds)")
    laser_freq = traits.Float(desc="Laser frequency (Hz)")

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.laser_gpio = ArduinoGPIO()

    def _start_trial(self):
        laser = DigitalWave(self.laser_gpio, pin=self.laser_gpio_pin)
        if self.laser_wave == "Constant":
            edges = [0]
            self.laser_duration = 0
        elif self.laser_wave == "Pulse":
            edges = [0, self.laser_duration]
        elif self.laser_wave == "Square wave":
            edges = DigitalWave.square_wave(self.laser_freq, self.laser_duration)
        laser.set_edges(edges, self.laser_first_edge)
        laser.start()
        super(LaserTrials, self)._start_trial()
