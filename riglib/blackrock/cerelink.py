'''
Client-side code to configure and receive neural data from the Blackrock
Neural Signal Processor (NSP).
'''

import sys
import time
from cerebus import cbpy
# from CereLink import cbpy  # old cbpy
from collections import namedtuple


SpikeEventData = namedtuple("SpikeEventData", ["chan", "unit", "ts", "arrival_ts"])
ContinuousData = namedtuple("ContinuousData", ["chan", "samples", "arrival_ts"])

class Connection(object):
    '''Here's a docstring'''

    def __init__(self):
        self.parameters = dict()
        self.parameters['inst-addr']   = '192.168.137.128'
        self.parameters['inst-port']   = 51001
        self.parameters['client-port'] = 51002

        if sys.platform == 'darwin':  # OS X
            print 'Using OS X settings for cbpy'
            self.parameters['client-addr'] = '255.255.255.255'
        else:  # linux
            print 'Using linux settings for cbpy'
            self.parameters['client-addr'] = '192.168.137.255'
            self.parameters['receive-buffer-size'] = 8388608

        self._init = False
    
    def connect(self):
        '''Open the interface to the NSP (or nPlay).'''

        print 'calling cbpy.open in cerelink.connect()'
        result, return_dict = cbpy.open(connection='default', parameter=self.parameters)
        print 'cbpy.open result:', result
        print 'cbpy.open return_dict:', return_dict
        print ''
        
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

        print 'calling cbpy.trial_config in cerelink.select_channels()'
        result, reset = cbpy.trial_config(buffer_parameter=buffer_parameter)
        print 'cbpy.trial_config result:', result
        print 'cbpy.trial_config reset:', reset
        print ''
    
    def start_data(self):
        '''Start the buffering of data.'''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        self.streaming = True

    def stop_data(self):
        '''Stop the buffering of data.'''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        print 'calling cbpy.trial_config in cerelink.stop()'
        result, reset = cbpy.trial_config(reset=False)
        print 'cbpy.trial_config result:', result
        print 'cbpy.trial_config reset:', reset
        print ''

        self.streaming = False

    def disconnect(self):
        '''Close the interface to the NSP (or nPlay).'''
        
        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")
        
        print 'calling cbpy.close in cerelink.disconnect()'
        result = cbpy.close()
        print 'result:', result
        print ''

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
        #    list of arrays [channel, {'timestamps':[unit0_ts, ..., unitN_ts], 'events':digital_events}]
        #        channel: integer, channel number (1-based)
        #        digital_events: array, digital event values for channel (if a digital or serial channel)
        #        unitN_ts: array, spike timestamps of unit N for channel (if an electrode channel));
        # '''

        sleep_time = 0.005

        while self.streaming:

            result, trial = cbpy.trial_event(reset=True)  # TODO -- check if result = 0?
            arrival_ts = time.time()

            for list_ in trial:
                chan = list_[0]
                for unit, unit_ts in enumerate(list_[1]['timestamps']):
                    for ts in unit_ts:
                        yield SpikeEventData(chan=chan, unit=unit, ts=ts, arrival_ts=arrival_ts)

            time.sleep(sleep_time)

            # TODO - sleep so that we don't call trial_event too often?

    def get_continuous_data(self):
        '''A generator which yields continuous data.'''

        # trial_continuous(instance = 0, reset=False):
        # ''' Trial continuous data.
        # Inputs:
        #    reset - (optional) boolean 
        #            set False (default) to leave buffer intact.
        #            set True to clear all the data and reset the trial time to the current time.
        #    instance - (optional) library instance number
        # Outputs:
        #    list of the form [channel, continuous_array]
        #        channel: integer, channel number (1-based)
        #        continuous_array: array, continuous values for channel)
        # '''

        while self.streaming:
            result, trial = cbpy.trial_continuous()
            arrival_ts = time.time()

            for list_ in trial:
                chan = list_[0]
                samples = list_[1]
                yield ContinuousData(chan=chan, samples=samples, arrival_ts=arrival_ts)

            # TODO - sleep so that we don't call trial_continuous too often?


if __name__ == "__main__":
    import csv
    import time
    import argparse
    parser = argparse.ArgumentParser(description="Collects plexnet data for a set amount of time")
    parser.add_argument("output", help="Output csv file")
    args = parser.parse_args()

    with open(args.output, "w") as f:
        csvfile = csv.DictWriter(f, SpikeEventData._fields)
        csvfile.writeheader()

        channels = [5, 6, 7, 8]

        conn = Connection()
        conn.connect()
        conn.select_channels(channels)
        conn.start_data() #start the data pump

        gen = conn.get_event_data()

        got_first = False

        start = time.time()
        while (time.time()-start) < 3:
            spike_event_data = gen.next()
            if not got_first and spike_event_data is not None:
                print spike_event_data
                got_first = True

            if spike_event_data is not None:
                csvfile.writerow(dict(spike_event_data._asdict()))

        #Stop the connection
        conn.stop_data()
        conn.disconnect()
