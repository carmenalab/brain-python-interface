'''Needs docs'''


from __future__ import division
import time
import numpy as np
import plexnet
from collections import Counter
#import logging
import os

class Spikes(object):
    update_freq = 40000
    dtype = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32), ("arrival_ts", np.float64)])

    def __init__(self, addr=("10.0.0.13", 6000), channels=None):
        log_filename = os.path.expandvars('$HOME/code/bmi3d/log/plexon_spikes.log')
        f = open(log_filename, 'a')
        f.write("Spikes source creating plexnet connection (Spikes.init, plexnet/__init__.py)\n")
        f.close()

        self.conn = plexnet.Connection(*addr)

        f = open(log_filename, 'a')
        f.write("Spikes source calling plexnet connect method (Spikes.init, plexnet/__init__.py)\n")
        f.close()

        self.conn.connect(256, waveforms=False, analog=False)

        f = open(log_filename, 'a')
        f.write("self.conn.connect finished\n")
        f.close()

        try:
            self.conn.select_spikes(channels)

            f = open(log_filename, 'a')
            f.write("channels selected\n\n")
            f.close()            
        except:
            print "Cannot run select_spikes method; old system?"

    def start(self):
        #logging.debug("Spikes object starting plexnet connection (Spikes.init, plexnet/__init__.py)")
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        self.conn.stop_data()
        self.conn.disconnect()

    def get(self):
        d = self.data.next()
        while d.type != 1:
            d = self.data.next()

        return np.array([(d.ts / self.update_freq, d.chan, d.unit, d.arrival_ts)], dtype=self.dtype)

class SimSpikes(object):
    update_freq = 65536
    dtype = np.dtype([("ts", np.float), ("chan", np.int), ("unit", np.int)])

    def __init__(self, afr=10, channels=600):
        self.rates = np.random.gamma(afr, size=channels)

    def start(self):
        self.wait_time = np.random.exponential(1/self.rates)
        
    def stop(self):
        pass

    def pause(self):
        self.start()

    def get(self):
        am = self.wait_time.argmin()
        time.sleep(self.wait_time[am])
        self.wait_time -= self.wait_time[am]
        self.wait_time[am] = np.random.exponential(1/self.rates[am])
        return np.array([(time.time()*1e6, am, 0)], dtype=self.dtype)

class PSTHfilter(object):
    def __init__(self, length, cells=None):
        self.length = length
        self.cells = cells

    def __call__(self, raw):
        if len(raw) < 1:
            return None

        data = raw[ (raw['ts'][-1] - raw['ts']) < self.length ]
        counts = Counter(data[['chan', 'unit']])
        if self.cells is not None:
            ret = np.array([counts[c] for c in self.cells])
            0/0
            return ret
        return counts

def test_filter():
    from riglib import source
    ds = source.DataSource(SimSpikes, channels=100)
    ds.start()
    ds.filter = PSTHfilter(100000, cells=zip(range(100), [0]*100))
    
    times = np.zeros(10000)
    for i in range(len(times)):
        times[i] = time.time()
        print ds.get(True)
        times[i] = time.time() - times[i]
        time.sleep(1/60.)
    return times
