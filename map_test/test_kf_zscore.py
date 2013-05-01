import numpy as np
import pickle
import socket
import struct
import time
from scipy.io import loadmat

import kfdecoder
import dsp

target_locations = loadmat('jeev_center_out_bmi_targets_post012813.mat')
center = target_locations['centerPos'].ravel()
horiz_min, vert_min = center - np.array([0.09, 0.09])
horiz_max, vert_max = center + np.array([0.09, 0.09])
bounding_box = (horiz_min, vert_min, horiz_max, vert_max)

decoder_fname = 'jeev041513_VFB_Kawf_B100_NS5_NU14_Z1_smoothbatch_smoothbatch_smoothbatch_smoothbatch_smoothbatch_smoothbatch.mat'
decoder = kfdecoder.load_from_mat_file(decoder_fname, bounding_box=bounding_box)

norm_decoder_fname = 'jeev042813_VFB_Kawf_B100_NS5_NU14_Z1.mat'
norm_decoder = kfdecoder.load_from_mat_file(norm_decoder_fname)

decoder.init_zscore(norm_decoder.mFR, norm_decoder.sdFR)

bmi_data = loadmat('jeev042813_BMI_K_1')
y = bmi_data['X_all']
x_kf = bmi_data['Yh_hat_all']

try:
    T = np.nonzero(np.isnan(y[0,:]))[0][0]
except:
    pass

y = y[:, :T]

spike_counts = np.zeros(y.shape)
for t in range(T):
    spike_counts[:,t] = y[:,t]*np.ravel(norm_decoder.sdFR/decoder.sdFR) + norm_decoder.mFR.ravel()

spike_counts = np.around(spike_counts, 1)
#spike_counts[np.abs(spike_counts) < 1e-10] = 0

units = decoder.units
n_neurons = units.shape[0]

ts_dtype = [('ts', float), ('chan', np.int32), ('unit', np.int32)]
T_sim = 5
x_kf_recon = np.zeros([5, T_sim])
for t in range(T_sim):
    obs = spike_counts[:,t]
    data_t = []
    for k in range(n_neurons):
        data_t += [(-1, units[k, 0], units[k,1]) for m in range(int(obs[k]))]

    data_t = np.array(data_t, dtype=ts_dtype)
    assert np.array_equal(decoder.bin_spikes(data_t), spike_counts[:,t])

    x_kf_recon[:,t] = decoder.decode(data_t)
    print y[:,t]
