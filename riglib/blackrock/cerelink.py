'''
Client-side code that uses cbpy to configure and receive neural data from the 
Blackrock Neural Signal Processor (NSP) (or nPlay).
'''

import sys
import time
from collections import namedtuple
try:
    from cerebus import cbpy
except ImportError:
    import warnings
    warnings.warn("Unable to import cerebus library. Check if is installed if using the Blackrock NeuroPort system")

SpikeEventData = namedtuple("SpikeEventData",
                            ["chan", "unit", "ts", "arrival_ts"])
ContinuousData = namedtuple("ContinuousData", 
                            ["chan", "samples", "arrival_ts"])

class Connection(object):
    '''
    A wrapper around a UDP socket which sends the Blackrock NeuroPort system commands and 
    receives data. Must run in a separte process (e.g., through `riglib.source`) 
    if you want to use it as part of a task (e.g., BMI control)
    '''
    debug = False
    def __init__(self):
        self.parameters = dict()
        self.parameters['inst-addr']   = '192.168.137.128'
        self.parameters['inst-port']   = 51001
        self.parameters['client-port'] = 51002

        self.channel_offset = 0  # used to be 4 -- some old bug with nPlay
        print('Using cbpy channel offset of:', self.channel_offset)

        if sys.platform == 'darwin':  # OS X
            print('Using OS X settings for cbpy')
            self.parameters['client-addr'] = '255.255.255.255'
        else:  # linux
            print('Using linux settings for cbpy')
            self.parameters['client-addr'] = '192.168.137.255'
            self.parameters['receive-buffer-size'] = 8388608

        self._init = False

        if self.debug:
            self.nsamp_recv = 0
            self.nsamp_last_print = 0
    
    def connect(self):
        '''Open the interface to the NSP (or nPlay).'''

        print('calling cbpy.open in cerelink.connect()')
        # try:
        #     result, return_dict = cbpy.open(connection='default')
        #     time.sleep(3)
        # except:
        result, return_dict = cbpy.open(connection='default', parameter=self.parameters)
        time.sleep(3)

        print('cbpy.open result:', result)
        print('cbpy.open return_dict:', return_dict)
        if return_dict['connection'] != 'Master':
            raise Exception
        print('')
        
        # return_dict = cbpy.open('default', self.parameters)  # old cbpy
        
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

        buffer_parameter = {'absolute': True}  # want absolute timestamps

        # ability to select desired channels not yet implemented in cbpy        
        # range_parameter = dict()
        # range_parameter['begin_channel'] = channels[0]
        # range_parameter['end_channel']   = channels[-1]

        print('calling cbpy.trial_config in cerelink.select_channels()')
        result, reset = cbpy.trial_config(buffer_parameter=buffer_parameter)
        print('cbpy.trial_config result:', result)
        print('cbpy.trial_config reset:', reset)
        print('')
    
    def start_data(self):
        '''Start the buffering of data.'''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        self.streaming = True

    def stop_data(self):
        '''Stop the buffering of data.'''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        print('calling cbpy.trial_config in cerelink.stop()')
        result, reset = cbpy.trial_config(reset=False)
        print('cbpy.trial_config result:', result)
        print('cbpy.trial_config reset:', reset)
        print('')

        self.streaming = False

    def disconnect(self):
        '''Close the interface to the NSP (or nPlay).'''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")
        
        print('calling cbpy.close in cerelink.disconnect()')
        result = cbpy.close()
        print('result:', result)
        print('')

        self._init = False
    
    def __del__(self):
        self.disconnect()

    def get_event_data(self):
        '''A generator that yields spike event data.'''

        sleep_time = 0

        while self.streaming:

            result, trial = cbpy.trial_event(reset=True)  # TODO -- check if result = 0?
            arrival_ts = time.time()

            for list_ in trial:
                chan = list_[0]
                for unit, unit_ts in enumerate(list_[1]['timestamps']):
                    for ts in unit_ts:
                        # blackrock unit numbers are 0-based where zero is unsorted unit
                        if unit == 0:
                            # Unsorted units are unit 10 (j)
                            un = 10
                        else:
                            un = unit
                        yield SpikeEventData(chan=chan-self.channel_offset, unit=un, ts=ts, arrival_ts=arrival_ts)

            time.sleep(sleep_time)


    def get_continuous_data(self):
        '''A generator that yields continuous data.'''

        sleep_time = 0

        while self.streaming:
            result, trial = cbpy.trial_continuous(reset=True)
            arrival_ts = time.time()

            for list_ in trial:

                if self.debug:
                    chan = list_[0]
                    samples = list_[1]
                    if chan == 8:
                        self.nsamp_recv += len(samples)
                        if self.nsamp_recv > self.nsamp_last_print + 2000:
                            print("cerelink.py: # received =", self.nsamp_recv)
                            self.nsamp_last_print = self.nsamp_recv

                yield ContinuousData(chan=list_[0],
                                     samples=list_[1],
                                     arrival_ts=arrival_ts)

            time.sleep(sleep_time)

