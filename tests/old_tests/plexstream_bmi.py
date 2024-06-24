import time
from riglib import source
from riglib.plexon import Spikes
import pickle

ds = source.DataSource(Spikes)
ds.start()
time.sleep(0.1)

T = 1000
data = [None]*T
for k in range(T):
    data[k] = ds.get()
    data_k = data[k]
    print(data_k[data_k['chan'] == 232])

    time.sleep(0.1)
ds.stop()

pickle.dump(data, open('plexstream_bmi_test.dat', 'wb'))
