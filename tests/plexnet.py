import time
import numpy as np
from riglib.plexon import plexnet

conn = plexnet.Connection("10.0.0.13", 6000)
conn.connect(256, waveforms=False, analog=False)

# Load a decoder and get the chnnels to filter for
import pickle
dec = pickle.load(open('/storage/decoders/cart20131006_01_BBcart20131006_02.pkl'))
channels = dec.units[:,0]

select_all = False 
if select_all:
    conn.select_spikes(waveforms=False, unsorted=True)
else:
    conn.select_spikes(channels=channels, waveforms=False, unsorted=True)

data = []
d = conn.get_data()
t = time.time()
conn.start_data()
while time.time() - t < 20:
    pak = d.next()
    if pak.type == 1:
        data.append((pak.ts, pak.chan, pak.unit, pak.arrival_ts))

conn.stop_data()
conn.disconnect()

data = np.array(data, dtype=[('ts', float), ('chan', np.int32), ('unit', np.int32), ('arrival_ts', np.float64)])
data['ts'] /= 40000

offset = 100
unique_data = data[offset:]
mean_offset = np.mean(unique_data['arrival_ts'] - unique_data['ts'])
jitter = unique_data['arrival_ts'] - unique_data['ts'] - mean_offset
abs_jitter = np.abs(jitter)
print len(abs_jitter)
print len(np.nonzero(abs_jitter < 0.005)[0])
print len(np.nonzero(abs_jitter < 0.010)[0])
