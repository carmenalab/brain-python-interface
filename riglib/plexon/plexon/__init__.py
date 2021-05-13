'''
Base code for 'bmi' feature (both spikes and field potentials) when using the plexon system
'''
import time
import numpy as np
from collections import Counter
import os
import array

from riglib.source import DataSourceSystem
from . import plexnet

PL_SingleWFType = 1
PL_ExtEventType = 4
PL_ADDataType   = 5

class Spikes(DataSourceSystem):
    '''
    Client for spike data streamed from plexon system, compatible with riglib.source.DataSource
    '''
    update_freq = 40000
    dtype = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32), ("arrival_ts", np.float64)])

    def __init__(self, addr=("127.0.0.1", 6000), channels=None):
        '''
        Constructor for plexon.Spikes

        Parameters
        ----------
        addr: tuple of length 2
            (IP address, UDP port)
        channels: optional, default = None
            list of channels (electrodes) from which to receive spike data

        Returns
        -------
        Spikes instance
        '''
        self.conn = plexnet.Connection(*addr)
        self.conn.connect(256, waveforms=False, analog=False)

        try:
            self.conn.select_spikes(channels)
        except:
            print("Cannot run select_spikes method; old system?")

    def start(self):
        '''
        Connect to the plexon server and start receiving data
        '''
        self.conn.start_data()

        # self.data is a generator (the result of self.conn.get_data() is a 'yield').
        # Calling 'self.data.next()' in the 'get' function pulls a new spike timestamp
        self.data = self.conn.get_data()

    def stop(self):
        '''
        Disconnect from the plexon server
        '''
        self.conn.stop_data()
        self.conn.disconnect()

    def get(self):
        '''
        Return a single spike timestamp/waveform. Must be polled continuously for additional spike data. The polling is automatically taken care of by riglib.source.DataSource
        '''
        d = next(self.data)
        while d.type != PL_SingleWFType:
            d = next(self.data)

        return np.array([(d.ts / self.update_freq, d.chan, d.unit, d.arrival_ts)], dtype=self.dtype)


class LFP(DataSourceSystem):
    '''
    Client for local field potential data streamed from plexon system, compatible with riglib.source.MultiChanDataSource
    '''
    update_freq = 1000.

    gain_digiamp = 1000.
    gain_headstage = 1.

    # like the Spikes class, dtype is the numpy data type of items that will go
    #   into the (multi-channel, in this case) datasource's ringbuffer
    # unlike the Spikes class, the get method below does not return objects of
    #   this type (this has to do with the fact that a potentially variable
    #   amount of LFP data is returned in d.waveform every time
    #   self.data.next() is called
    dtype = np.dtype('float')

    def __init__(self, addr=("127.0.0.1", 6000), channels=None, chan_offset=512):
        '''
        Constructor for plexon.LFP

        Parameters
        ----------
        addr : tuple of length 2
            (IP address, UDP port)
        channels : optional, default = None
            list of channels (electrodes) from which to receive spike data
        chan_offset : int, optional, default=512
            Indexing offset from the first LFP channel to the indexing system used by the OPX system

        Returns
        -------
        plexon.LFP instance
        '''
        self.conn = plexnet.Connection(*addr)
        self.conn.connect(256, waveforms=False, analog=True)

        # for OPX system, field potential (FP) channels are numbered 513-768
        self.chan_offset = chan_offset
        channels_offset = [c + self.chan_offset for c in channels]
        try:
        	self.conn.select_continuous(channels_offset)
        except:
            print("Cannot run select_continuous method")

    def start(self):
        '''
        Connect to the plexon server and start receiving data
        '''
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        '''
        Disconnect from the plexon server
        '''
        self.conn.stop_data()
        self.conn.disconnect()

    def get(self):
        '''
        Get a new LFP sample/block of LFP samples from the
        '''
        d = next(self.data)
        while d.type != PL_ADDataType:
            d = next(self.data)

        # values are in currently signed integers in the range [-2048, 2047]
        # first convert to float
        waveform = np.array(d.waveform, dtype='float')

        # convert to units of mV
        waveform = waveform * 16 * (5000. / 2**15) * (1./self.gain_digiamp) * (1./self.gain_headstage)

        return (d.chan-self.chan_offset, waveform)


class Aux(DataSourceSystem):
    '''
    Client for auxiliary analog data streamed from plexon system, compatible with riglib.source.MultiChanDataSource
    '''
    update_freq = 1000.

    gain_digiamp = 1.
    gain_headstage = 1.

    # see comment above
    dtype = np.dtype('float')

    def __init__(self, addr=("127.0.0.1", 6000), channels=None, chan_offset=768):
        '''
        Constructor for plexon.Aux

        Parameters
        ----------
        addr : tuple of length 2
            (IP address, UDP port)
        channels : optional, default = None
            list of channels (electrodes) from which to receive spike data
        chan_offset : int, optional, default=768
            Indexing offset from the first Aux channel to the indexing system used by the OPX system

        Returns
        -------
        plexon.Aux instance
        '''
        self.conn = plexnet.Connection(*addr)
        self.conn.connect(256, waveforms=False, analog=True)

        # for OPX system, the 32 auxiliary input (AI) channels are numbered 769-800
        self.chan_offset = chan_offset

        channels_offset = [c + self.chan_offset for c in channels]
        try:
            self.conn.select_continuous(channels_offset)
        except:
            print("Cannot run select_continuous method")

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = next(self.data)
        while d.type != PL_ADDataType:
            d = next(self.data)

        # values are in currently signed integers in the range [-2048, 2047]
        # first convert to float
        waveform = np.array(d.waveform, dtype='float')

        # convert to units of mV
        waveform = waveform * 16 * (5000. / 2**15) * (1./self.gain_digiamp) * (1./self.gain_headstage)

        return (d.chan-self.chan_offset, waveform)

