import time
from riglib import source
from riglib.ecube import Broadband, make
import numpy as np

DURATION = 5

ecube = make(Broadband, headstages=[7], channels=[(1,1)])
ds = source.DataSource(ecube)
ds.start()
time.sleep(DURATION)
data = ds.get()
ds.stop()

print("Received {} packets in {} seconds ({} Hz)".format(data.shape[0], DURATION, data.shape[0]/DURATION))
ts = [d['timestamp'] for d in data]
print("First timestamp: {} ns ({:.5f} s)\tLast timestamp {} ns ({:.5f} s)".format(
    ts[0], ts[0]/1e9, ts[-1], ts[-1]/1e9
))
duration = (ts[-1]-ts[0])/1e9
samples = [d['data'].shape[0] for d in data]
print("Calculated duration: {:.5f} s, at {:.2f} Hz packet frequency and {} Hz sampling rate".format(
    duration, data.shape[0]/duration, np.sum(samples[:-1])/duration))
print("For reference, the ecube datasource is meant to have a sampling rate of {:.2f} Hz".format(ecube.update_freq))

