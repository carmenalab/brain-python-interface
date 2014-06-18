'''
Base code for 'bmi' feature (both spikes and fields) when using the plexon system
'''


from __future__ import division
import time
import numpy as np
import plexnet
import plexnet_softserver_oldfiles
from collections import Counter
import os
import array

PL_SingleWFType = 1
PL_ExtEventType = 4
PL_ADDataType   = 5

class Spikes(object):
    ''' Docstring '''
    update_freq = 40000
    dtype = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32), ("arrival_ts", np.float64)])

    def __init__(self, addr=("10.0.0.13", 6000), channels=None):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.conn = plexnet.Connection(*addr)
        self.conn.connect(256, waveforms=False, analog=False)

        try:
            self.conn.select_spikes(channels)
        except:
            print "Cannot run select_spikes method; old system?"

    def start(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.conn.stop_data()
        self.conn.disconnect()

    def get(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        d = self.data.next()
        while d.type != PL_SingleWFType:
            d = self.data.next()

        return np.array([(d.ts / self.update_freq, d.chan, d.unit, d.arrival_ts)], dtype=self.dtype)


class LFP(object):
    ''' Docstring '''
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

    def __init__(self, addr=("10.0.0.13", 6000), channels=None, chan_offset=512):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.conn = plexnet.Connection(*addr)
        self.conn.connect(256, waveforms=False, analog=True)
        # self.conn = plexnet_softserver_oldfiles.Connection(*addr)
        # self.conn.connect(192, waveforms=False, analog=True)

        # for OPX system, field potential (FP) channels are numbered 513-768
        self.chan_offset = chan_offset
        channels_offset = [c + self.chan_offset for c in channels]
        try:
        	self.conn.select_continuous(channels_offset)
        except:
            print "Cannot run select_continuous method"

    def start(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.conn.stop_data()
        # self.conn.disconnect()  # TODO -- Connection.__del__ calls disconnect too, only call this once?

    def get(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        d = self.data.next()
        while d.type != PL_ADDataType:
            d = self.data.next()

        # values are in currently signed integers in the range [-2048, 2047]
        # first convert to float
        waveform = np.array(d.waveform, dtype='float')

        # convert to units of mV
        waveform = waveform * 16 * (5000. / 2**15) * (1./self.gain_digiamp) * (1./self.gain_headstage)

        return (d.chan-self.chan_offset, waveform)

# TODO -- copied from LFP class above -- needs to be modified
class Aux(object):
    ''' Docstring '''
    update_freq = 1000.

    gain_digiamp = 1.
    gain_headstage = 1.

    # see comment above
    dtype = np.dtype('float')

    def __init__(self, addr=("10.0.0.13", 6000), channels=None, chan_offset=768):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.conn = plexnet.Connection(*addr)
        self.conn.connect(256, waveforms=False, analog=True)

        # for OPX system, the 32 auxiliary input (AI) channels are numbered 769-800
        self.chan_offset = chan_offset

        channels_offset = [c + self.chan_offset for c in channels]
        try:
            self.conn.select_continuous(channels_offset)
        except:
            print "Cannot run select_continuous method"

    def start(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.conn.stop_data()

    def get(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        d = self.data.next()
        while d.type != PL_ADDataType:
            d = self.data.next()

        # values are in currently signed integers in the range [-2048, 2047]
        # first convert to float
        waveform = np.array(d.waveform, dtype='float')

        # convert to units of mV
        waveform = waveform * 16 * (5000. / 2**15) * (1./self.gain_digiamp) * (1./self.gain_headstage)

        return (d.chan-self.chan_offset, waveform)


class SimSpikes(object):
    ''' Docstring '''
    update_freq = 65536
    dtype = np.dtype([("ts", np.float), ("chan", np.int), ("unit", np.int)])

    def __init__(self, afr=10, channels=600):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.rates = np.random.gamma(afr, size=channels)

    def start(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.wait_time = np.random.exponential(1/self.rates)
        
    def stop(self):
        pass

    def pause(self):
        self.start()

    def get(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        am = self.wait_time.argmin()
        time.sleep(self.wait_time[am])
        self.wait_time -= self.wait_time[am]
        self.wait_time[am] = np.random.exponential(1/self.rates[am])
        return np.array([(time.time()*1e6, am, 0)], dtype=self.dtype)

class PSTHfilter(objec
    ''' Docstring '''t):
    def __init__(self, length, cells=None):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.length = length
        self.cells = cells

    def __call__(self, raw):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if len(raw) < 1:
            return None

        data = raw[ (raw['ts'][-1] - raw['ts']) < self.length ]
        counts = Counter(data[['chan', 'unit']])
        if self.cells is not None:
            ret = np.array([counts[c] for c in self.cells])
            0/0
            return ret
        return counts

def test_filter(update_rate=60.):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    from riglib import source
    ds = source.DataSource(SimSpikes, channels=100)
    ds.start()
    ds.filter = PSTHfilter(100000, cells=zip(range(100), [0]*100))
    
    times = np.zeros(10000)
    for i in range(len(times)):
        times[i] = time.time()
        print ds.get(True)
        times[i] = time.time() - times[i]
        time.sleep(1/update_rate)
    return times
