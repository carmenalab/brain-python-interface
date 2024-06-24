import time
import numpy as np
from riglib.plexon import plexnet
import matplotlib.pyplot as plt
from riglib.experiment import traits

conn = plexnet.Connection("10.0.0.13", 6000)
conn.connect(256, waveforms=False, analog=False)

# Load a decoder and get the chnnels to filter for
import pickle
dec = pickle.load(open('/storage/decoders/cart20131006_01_BBcart20131006_02.pkl'))


class SpikeData(traits.HasTraits):
    '''Stream neural spike data from the Plexon system'''
    plexon_channels = None

    def init(self):
        from riglib import plexon, source
        self.neurondata = source.DataSource(plexon.Spikes, channels=self.plexon_channels)
        try:
            super(SpikeData, self).init()
        except:
            print("SpikeData: running without a task")
    
    def run(self):
        self.neurondata.start()


self = SpikeData()
self.init()
self.run()

N = 2000
update_rate = 1./60
data_ls = [None]*N
for k in range(N):
    data_ls[k] = self.neurondata.get()
    time.sleep(update_rate)
self.neurondata.stop()

## channels = dec.units[:,0]
## select_all = False 
## if select_all:
##     conn.select_spikes(waveforms=False, unsorted=True)
## else:
##     conn.select_spikes(channels=channels, waveforms=False, unsorted=True)
## 
## data = []
## d = conn.get_data()
## t = time.time()
## conn.start_data()
## while time.time() - t < 20:
##     pak = d.next()
##     if pak.type == 1:
##         data.append((pak.ts, pak.chan, pak.unit, pak.arrival_ts))
## 
## conn.stop_data()
## conn.disconnect()
## 
## data = np.array(data, dtype=[('ts', float), ('chan', np.int32), 
##                              ('unit', np.int32), ('arrival_ts', np.float64)])
## data['ts'] /= 40000

# Ignore first 0.5 of data
offset_time = 1.
offset_bins = int(1./update_rate)
data = np.hstack(data_ls)
unique_data = np.hstack(data_ls[offset_bins:])
#offset = np.nonzero(data['arrival_ts'] - data['arrival_ts'][0] > 0.5)[0][0]
#print "Offset from startup stuff from plexnet", offset
#unique_data = data[offset:]
mean_offset = np.mean(unique_data['arrival_ts'] - unique_data['ts'])
jitter = unique_data['arrival_ts'] - unique_data['ts'] - mean_offset
abs_jitter = np.abs(jitter)
print(len(abs_jitter))
print(len(np.nonzero(abs_jitter < 0.005)[0]))
print(len(np.nonzero(abs_jitter < 0.010)[0]))
print("Observed channels with units")
print(np.unique(data['chan']))

plt.close('all')
plt.figure()
plt.subplot(2,1,1)
plt.hold(True)
plt.plot(unique_data['ts'] - unique_data['ts'][0])
plt.plot(unique_data['arrival_ts'] - unique_data['arrival_ts'][0])
plt.subplot(2,1,2)
plt.hold(True)
plt.plot(data['ts'] - data['ts'][0])
plt.plot(data['arrival_ts'] - data['arrival_ts'][0])

plt.show()
