import time
import numpy as np
from riglib.plexon import plexnet

conn = plexnet.Connection("10.0.0.13", 6000)
conn.connect(256, waveforms=False, analog=False)
#conn.select_spikes()

data = []
d = conn.get_data()
t = time.time()
conn.start_data()
while time.time() - t < 10:
    pak = d.next()
    if pak.type == 1:
        data.append((pak.ts, pak.chan, pak.unit))

conn.stop_data()
conn.disconnect()

data = np.array(data, dtype=[('ts', float), ('chan', np.int32), ('unit', np.int32)])
data['ts'] /= 40000
