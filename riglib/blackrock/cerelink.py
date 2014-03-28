'''
Client-side code to configure and receive neural data from the Blackrock
Neural Signal Processor (NSP).
'''

import time
import cerebus.cbpy as cbpy
from collections import namedtuple


SpikeEventData = namedtuple("SpikeEventData", ["chan", "unit", "ts", "arrival_ts"])
ContinuousData = namedtuple("ContinuousData", ["chan", "samples", "ts", "arrival_ts"])

class Connection(object):
    '''Here's a docstring'''

    def __init__(self):
        self.parameters = dict()
        self.parameters['inst-addr']   = '192.168.137.128'
        self.parameters['inst-port']   = 51001
        self.parameters['client-addr'] = '192.168.137.255'
        self.parameters['client-port'] = 51002
        self.parameters['receive-buffer-size'] = 8388608

        self._init = False
    
    def connect(self):
        '''Open the interface to the NSP (or nPlay).'''

        result, return_dict = cbpy.open(connection='default', parameter=self.parameters)
        
        self._init = True
        
    def select_channels(self, channels):
        '''Sets the channels on which to receive event/continuous data.

        Parameters
        ----------
        channels : array_like
            A sorted list of channels on which you want to receive data.
        '''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")
        
        range_parameter = dict()
        range_parameter['begin_channel'] = channels[0]
        range_parameter['end_channel']   = channels[-1]

        result, reset = cbpy.trial_config(range_parameter=range_parameter)
    
    def start_data(self):
        '''Start the buffering of data.'''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        self.streaming = True

    def stop_data(self):
        '''Stop the buffering of data.'''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        cbpy.trial_config(reset=False)
        self.streaming = False

    def disconnect(self):
        '''Close the interface to the NSP (or nPlay).'''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")
        
        cbpy.close()
        self._init = False
    
    def __del__(self):
        self.disconnect()

    def get_event_data(self):
        '''A generator which yields spike event data.'''

        # trial_event(instance = 0, reset=False):
        # '''
        # Trial spike and event data.
        # Inputs:
        #    reset - (optional) boolean 
        #            set False (default) to leave buffer intact.
        #            set True to clear all the data and reset the trial time to the current time.
        #    instance - (optional) library instance number
        # Outputs:
        #    list of arrays [channel, digital_events] or [channel, unit0_ts, ..., unitN_ts]
        #        channel: integer, channel number (1-based)
        #        digital_events: array, digital event values for channel (if a digital or serial channel)
        #        unitN_ts: array, spike timestamps of unit N for channel (if an electrode channel));
        # '''

        while self.streaming:
            array_list = cbpy.trial_event()
            arrival_ts = time.time()

            for arr in array_list:
                chan = arr[0]
                for unit, unit_ts in enumerate(arr[1:]):
                    for ts in unit_ts:
                        yield SpikeEventData(chan=chan, unit=unit, ts=ts, arrival_ts=arrival_ts)

            # TODO - some kind of pause/sleep?

    def get_continuous_data(self):
        '''A generator which yields continuous data.'''

        while self.streaming:
            # make call to cbpy.trial_continuous()
            arrival_ts = time.time()

            # yield ContinuousData(chan=chan, samples=samples, ts=ts, arrival_ts=arrival_ts) in a loop

            # TODO - some kind of pause/sleep?


