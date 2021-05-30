import numpy as np
import time
import scipy.io as sio

from ismore import brainamp_channel_lists
from ..riglib.brainamp.rda import *



fs = 1000
channels = brainamp_channel_lists.emg_eog2_eeg

total_time = 120  # how many secs of data to receive and save

n_samples = 2 * fs * total_time  # allocate twice as much space as expected
n_chan = len(channels)


DATA = np.zeros((n_chan, n_samples))
idxs = np.zeros(n_chan, int)

chan_to_row = dict()
for row, chan in enumerate(channels):
    chan_to_row[chan] = row

emgdata_obj = EMGData()
emgdata_obj.start()

start_time = time.time()

while (time.time() - start_time) < total_time:
    chan, data = emgdata_obj.get()

    row = chan_to_row[chan]
    idx = idxs[row]

    DATA[row, idx] = data['data']
    idxs[row] += 1


save_dict = {'data': DATA}
sio.matlab.savemat('brainamp_data.mat', save_dict)
