#!/usr/bin/python
'''
Test case for PPFDecoder
'''
import numpy as np
from scipy.io import loadmat, savemat
from riglib.bmi import sim_neurons
import matplotlib.pyplot as plt
from riglib.bmi import ppfdecoder, train
import plotutil

plt.close('all')


# 1) get CT kinematics from Paco
from scipy.io import loadmat
data = loadmat('/Users/sgowda/code/bmi3d/tests/ppf/paco_hand_kin_200hz.mat')
hand_vel = data['hand_vel']*100 # convert to cm/s

hand_vel = np.vstack([hand_vel, np.ones(hand_vel.shape[1])])

# 2) create a fake beta
const = -1.6
alpha = 0.04
n_cells = 30
angles = np.linspace(0, 2*np.pi, n_cells)
beta = np.zeros([n_cells, 7])
beta[:,-1] = -1.6
beta[:,3] = np.cos(angles)*alpha
beta[:,5] = np.sin(angles)*alpha
beta = np.vstack([np.cos(angles)*alpha, np.sin(angles)*alpha, np.ones_like(angles)*const]).T

# create the encoder
dt = 0.005
encoder = sim_neurons.PointProcessEnsemble(beta, dt)

# simulate spike counts
T = 10000 #hand_vel.shape[1]
spike_counts = np.zeros([n_cells, T])
for t in range(T):
    spike_counts[:,t] = encoder(hand_vel[:,t])


savemat('sample_spikes_and_kinematics.mat', dict(spike_counts=spike_counts))

# run the decoder
beta_full = np.zeros([n_cells, 7])
beta_full[:,[3,5,6]] = beta
dec = train._train_PPFDecoder_sim_known_beta(beta_full, encoder.units, dt=dt, dist_units='m')

dec_output = np.zeros([7, T])
for t in range(T):
    dec_output[:,t] = dec.predict(spike_counts[:,t])

plt.figure()
axes = plotutil.subplots2(2, 1)
axes[0,0].plot(dec_output[3,:])
axes[0,0].plot(hand_vel[0,:])
axes[1,0].plot(dec_output[5,:])
axes[1,0].plot(hand_vel[1,:])
plotutil.set_axlim(axes, [0, T], axis='x')
plt.show()
plt.draw()

# N = 10000
# fname ='sample_spikes_and_kinematics_%d.mat' % N 
# data = loadmat(fname)
# truedata = loadmat('/Users/sgowda/bmi/workspace/adaptive_ppf/ppf_test_case_matlab_output.mat')
# x_est = truedata['x_est']
# X = data['hand_vel'].T

# beta = data['beta']
# beta = np.vstack([beta[1:, :], beta[0,:]]).T
# n_neurons = beta.shape[0]

# n_iter = X.shape[0]
# dt = 0.005

# encoder = sim_neurons.load_ppf_encoder_2D_vel_tuning(fname, dt=dt)

# states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
# decoding_states = ['hand_vx', 'hand_vz', 'offset'] 
# beta_dec = train.inflate(beta, decoding_states, states, axis=1)
# dec = train._train_PPFDecoder_sim_known_beta(beta_dec, encoder.units, dt=dt, dist_units='m')

# spike_counts = np.zeros([n_iter, n_neurons])
# decoded_output_new = np.zeros([7, n_iter])
# for n in range(1, n_iter):
#     spike_counts[n-1, :] = encoder(X[n,:])
#     decoded_output_new[:, n-1] = dec.predict(spike_counts[n-1 ,:])

# print "Python sim spikes matches MATLAB's: %s" % np.array_equal(spike_counts, data['spike_counts'])

# print np.max(np.abs(x_est[0,:n_iter:dec.n_subbins] - decoded_output_new[3,::dec.n_subbins]))
# print np.max(np.abs(x_est[0,:n_iter-1] - decoded_output_new[3,:-1]))

# plt.figure()
# plt.hold(True)
# plt.plot(decoded_output_new[3,:n_iter], label='pyth')
# plt.plot(x_est[0,:n_iter], label='matlab')
# plt.plot(X[:n_iter, 0], label='handvel')
# plt.legend()
# plt.show()
