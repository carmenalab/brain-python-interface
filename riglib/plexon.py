import numpy as np
import plexnet

class Spikes(object):
    update_freq = 65536
    dtype = np.dtype([("ts", np.uint64), ("chan", np.uint16), ("unit", np.uint16)])
    
    def __init__(self, addr=("10.0.0.13", 6000)):
        self.conn = plexnet.Connection(*addr)
        self.conn.connect(256, waveforms=False)
        self.conn.select_spikes(waveforms=False)

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
