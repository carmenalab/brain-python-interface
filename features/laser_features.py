'''
Laser delivery features
'''

import time
import traceback
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
    qwalor_trigger_dch = traits.Int(9, desc="Digital channel (0-index) recording laser trigger")
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
            time.sleep(3) # some extra time to make sure the lasers are initialized

        except Exception as e:
            traceback.print_exc()
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
    qwalor_ch1_trigger_dch = traits.Int(8, desc="Digital channel (0-index) recording laser ch1 trigger")
    qwalor_ch2_trigger_dch = traits.Int(9, desc="Digital channel (0-index) recording laser ch2 trigger")
    qwalor_ch3_trigger_dch = traits.Int(10, desc="Digital channel (0-index) recording laser ch3 trigger")
    qwalor_ch4_trigger_dch = traits.Int(11, desc="Digital channel (0-index) recording laser ch4 trigger")
    qwalor_ch1_sensor_ach = traits.Int(15, desc="Analog channel (0-index) recording laser ch1 power")
    qwalor_ch2_sensor_ach = traits.Int(16, desc="Analog channel (0-index) recording laser ch2 power")
    qwalor_ch3_sensor_ach = traits.Int(17, desc="Analog channel (0-index) recording laser ch3 power")
    qwalor_ch4_sensor_ach = traits.Int(18, desc="Analog channel (0-index) recording laser ch4 power")
    
    hidden_traits = ['qwalor_ch1_sensor_ach', 'qwalor_ch2_sensor_ach', 'qwalor_ch3_sensor_ach', 'qwalor_ch4_sensor_ach',
                     'qwalor_ch1_trigger_dch', 'qwalor_ch2_trigger_dch', 'qwalor_ch3_trigger_dch', 'qwalor_ch4_trigger_dch']

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

            time.sleep(3) # some extra time to make sure the lasers are initialized
            self.qwalor_laser_status = 'ok'
            
        except Exception as e:
            traceback.print_exc()
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

class LaserState(traits.HasTraits):
    '''
    Trigger lasers to stimulate at the beginning of a given state. Make sure the state duration is 
    longer than the total stimulation duration otherwise the stim may extend into other states.
    '''

    laser_trigger_state = traits.String("wait", desc="State machine state that triggers laser")
    laser_stims_per_trial = traits.Int(1, desc="Number of stimulations per laser per trial")
    laser_power = traits.List([1.,], desc="Laser power (between 0 and 1) for each active laser")
    laser_pulse_width = traits.List([0.005,], desc="List of possible pulse widths in seconds")
    laser_poisson_mu = traits.Float(0.5, desc="Mean duration between laser stimulations (s)")

    hidden_traits = ['laser_trigger_state']

    def __init__(self, *args, **kwargs):
        self.laser_threads = []
        super().__init__(*args, **kwargs)
    
    def run(self):
        if not (hasattr(self, 'lasers') and len(self.lasers) > 0):
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write("No laser feature enabled, cannot init LaserState")
            self.termination_err.seek(0)
            self.state = None
        super().run() 

    def start_state(self, state):
        super().start_state(state)
        if state != self.laser_trigger_state:
            return
        
        wait_time = 0
        self.laser_waves = []
        for idx in range(len(self.lasers)):
            laser = self.lasers[idx]

            width_idx = np.random.choice(len(self.laser_pulse_width))
            width = self.laser_pulse_width[width_idx]
            power = self.laser_power[idx]
            laser.set_power(power)

            for n in range(self.laser_stims_per_trial):

                # Trigger digital wave
                wave = DigitalWave(laser, mask=1<<laser.port)
                wave.set_edges([wait_time, wait_time+width], True)
                wave.start()
                self.laser_waves.append(wave)

                # Make the next pulse come after a delay
                delay = max(np.random.exponential(self.laser_poisson_mu), 2*width)
                wait_time += delay
                print(wait_time)

        print(self.laser_waves)
