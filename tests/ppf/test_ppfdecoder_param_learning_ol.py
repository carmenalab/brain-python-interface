#!/usr/bin/python
'''
Test trying to learn the beta parameters using SB and the continuous relearning
methods
'''
import numpy as np
from scipy.io import loadmat, savemat
from riglib.bmi import sim_neurons
import matplotlib.pyplot as plt
from riglib.bmi import ppfdecoder, train, clda
import plot

reload(ppfdecoder)
reload(sim_neurons)
reload(train)
reload(clda)
plt.close('all')

N = 168510.
fname = 'sample_spikes_and_kinematics_%d.mat' % N 
data = loadmat(fname)
truedata = loadmat('/Users/sgowda/bmi/workspace/adaptive_ppf/ppf_test_case_matlab_output.mat')
X = data['hand_vel'].T

beta = data['beta']
beta = np.vstack([beta[1:, :], beta[0,:]]).T
n_neurons = beta.shape[0]
dt = truedata['T_loop'][0,0]

encoder = sim_neurons.load_ppf_encoder_2D_vel_tuning(fname, dt=dt)
states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
decoding_states = ['hand_vx', 'hand_vz', 'offset'] 

# initialze estimate of beta
beta_est = beta.copy()
beta_est[:,0:2] = 0
beta_est = train.inflate(beta_est, decoding_states, states, axis=1)
decoder = train._train_PPFDecoder_sim_known_beta(beta_est, encoder.units, dt=dt, dist_units='m')

# Initialize learner and updater
batch_time = 60.
batch_size = batch_time/dt
half_life = 120.
rho = np.exp(np.log(0.5) / (half_life/batch_time))

learner = clda.BatchLearner(batch_size)
updater = clda.PPFSmoothbatchSingleThread()
updater.rho = rho

## RUN 
n_iter = X.shape[0]
spike_counts = data['spike_counts']
beta_hist = []
for n in range(1, n_iter):
    if n % batch_size == 0: print n
    int_kin = np.hstack([np.zeros(3), X[n,0], 0, X[n,1], 1])

    learner(spike_counts[n-1, :].reshape(-1,1), int_kin)

    if learner.is_full():
        # calc beta est from batch
        int_kin_batch, spike_counts_batch = learner.get_batch()
        beta_hist.append(decoder.filt.C)
        new_params = updater.calc(int_kin_batch, spike_counts_batch, rho, decoder)
        decoder.update_params(new_params)
        
# Plot results
beta_hist = map(lambda x: np.array(x), beta_hist)
beta_hist = np.dstack(beta_hist).transpose([2,0,1])

plt.figure()
axes = plot.subplots(5, 4, return_flat=True, hold=True)
for k in range(n_neurons):
    axes[k].plot(beta_hist[:,k,3])

plt.figure()
axes = plot.subplots(5, 4, return_flat=True, hold=True)
for k in range(n_neurons):
    axes[k].plot(beta_hist[:,k,5])

plt.figure()
axes = plot.subplots(2, 1, return_flat=True, hold=True)
axes[0].plot(beta[:,0])
axes[0].plot(decoder.filt.C[:,3])
axes[1].plot(beta[:,1])
axes[1].plot(decoder.filt.C[:,5])
plt.show()
