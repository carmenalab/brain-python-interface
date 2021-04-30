'''
This module appears to be deprecated.....
'''

from . import plexnet
from riglib.source import DataSource

class _PlexCont(object):
    def __init__(self, addr=("10.0.0.13", 6000), channels=None):
        self.conn = plexnet.Connection(*addr)
        self.conn.connect(256, waveforms=False, analog=True)
        self.conn.select_continuous(channels)

    def start(self):
        self.conn.start_data()
        self.data = self.conn.get_data()

    def stop(self):
        self.conn.stop_data()

    def get(self):
        d = next(self.data)
        while d.type != 5:
            d = next(self.data)

        return np.array([(d.ts, d.chan, d.unit)], dtype=self.dtype)

class Continuous(DataSource):
    def __init__(self, channels=None):
        self.source = _PlexCont(channels)

    def _get(self, system):
        pass