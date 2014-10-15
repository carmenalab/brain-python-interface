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
"""

class StimulationPulse(traits.HasTraits):
    '''
    Class for a single stimulation pulse
    '''
    def __init__(self,*args, **kwargs)


class TTLStimulation(StimulationPulse, traits.HasTraits):
    '''During the stimulation phase, send a timed TTL pulse to the Master-8 stimulator'''
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

    def _start_stimulation(self):
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
        super(TTLStimulation, self)._start_stimulation()
        subdevice = 0
        write_mask = 0x800000
        val = 0x800000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        self.stimulation_start = self.get_time() - self.start_time

    def _test_stimulation_end(self, ts):
        return (ts - self.stimulation_start) > self.stimulation_time

    def _end_stimulation(self):
        '''
        After the stimulation state has elapsed, make sure stimulation is off 

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        import comedi
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, 0x000000, base_channel)