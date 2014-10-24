'''
Code for the triggering stimulation from the Master-8 voltage-programmable stimulator
'''

import time
import tempfile
import random
import traceback
import numpy as np
import fnmatch
import os

from riglib import calibrations, bmi

from riglib.bmi import extractor

from . import traits, experiment

import os
import subprocess

import time

###### CONSTANTS
sec_per_min = 60

"""
Define one class to send TTL pulse for one cycle of stimulation (StimulationPulse).  Define second class to implement this
at a fixed rate.

Left off at line 79.  Need to figure out how we re-trigger starting the stimulus pulse train again.
"""



class TTLStimulation(StimulationPulse, traits.HasTraits):
    '''During the stimulation phase, send a timed TTL pulse to the Master-8 stimulator'''
    
    status = dict(
        pulse = dict(pulse_end="interpulse_period", stop=None),
        interpulse_period = dict(another_pulse="pulse", pulse_train_end="off", stop=None),
        pulse_off= dict(next_pulse_train="pulse", stop=None)
    )

    pulse_count = 0  #initializing number of pulses that have occured
    number_of_pulses = int(self.stimulation_period_length*self.stimulation_frequency)   # total pulses during a stimulation pulse train

    def __init__(self, *args, **kwargs):
        '''
        Constructor for TTLStimulation

        Parameters
        ----------
        pulse_device: string
            Path to the NIDAQ device used to generate the solenoid pulse
        args, kwargs: optional positional and keyword arguments to be passed to parent constructor
            None necessary

        Returns
        -------
        TTLStimulation instance
        '''
        import comedi
        self.com = comedi.comedi_open('/dev/comedi0')
        super(TTLStimulation, self).__init__(*args, **kwargs)

    def init(self):
        super(TTLStimulation, self).init()

    #### TEST FUNCTIONS ####

    def _test_pulse_end(self, ts):
        #return true if time has been longer than the specified pulse duration
        pulse_length = self.stimulation_pulse_length*1e-6
        return ts>=pulse_length

    def _test_another_pulse(self,ts):
        return pulse_count < number_of_pulses

    def _test_pulse_train_end(self,ts):
        return pulse_count > number_of_pulses

    def _test_next_pulse_train(self,ts):
        return self.enter_stimulation_state

    #### STATE FUNCTIONS ####

    def _start_pulse(self):
        '''
        At the start of the stimulation state, send TTL pulse

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        import comedi
        super(TTLStimulation, self)._start_pulse()
        subdevice = 0
        write_mask = 0x800000
        val = 0x800000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        #self.stimulation_start = self.get_time() - self.start_time

    def _end_pulse(self):
        '''
        After the stimulation state has elapsed, make sure stimulation is off 

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        pass

    def _start_interpulse_period(self):
        super(TTLStimulation, self)._start_interpulse_period()
        subdevice = 0
        write_mask = 0x800000
        val = 0x800000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)

    def _end_interpulse_period(self):
        super(TTLStimulation, self)._end_interpulse_period()
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)

    def _start_pulse_off(self):
        super(TTLStimulation, self)._start_pulse_off()
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)

    def _end_pulse_off(self):
        pass