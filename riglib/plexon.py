import time
import numpy as np
import plexnet

class Spikes(object):
    update_freq = 65536
    dtype = np.dtype([("ts", np.uint64), ("chan", np.uint16), ("unit", np.uint16)])
    
    def __init__(self, addr=("10.0.0.13", 6000), channels=None):
        self.conn = plexnet.Connection(*addr)
        self.conn.connect(256, waveforms=False)
        self.conn.select_spikes(channels, waveforms=False)

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = self.data.next()
        while d.type != 5:
            d = self.data.next()

        return np.array([(d.ts, d.chan, d.unit)], dtype=self.dtype)

class SimSpikes(object):
    update_freq = 65536
    dtype = np.dtype([("ts", np.uint64), ("chan", np.uint16), ("unit", np.uint16)])

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
    def __init__(self, length, cells):
        self.length = length
        self.cells = cells

    def __call__(self, raw):
        data = raw[ (raw['ts'][-1] - raw['ts']) < self.length ]
        counts = []
        for chan, unit in self.cells:
            num = np.sum(np.logical_and(data['chan'] == chan, data['unit'] == unit))
            counts.append(num)
            
        return counts
