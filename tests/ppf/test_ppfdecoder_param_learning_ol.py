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
plt.close('all')

N = 168510.
fname ='sample_spikes_and_kinematics_%d.mat' % N 
data = loadmat(fname)
truedata = loadmat('/Users/sgowda/bmi/workspace/adaptive_ppf/ppf_test_case_matlab_output.mat')
x_est = truedata['x_est']
X = data['hand_vel'].T

beta = data['beta']
beta = np.vstack([beta[1:, :], beta[0,:]]).T
n_neurons = beta.shape[0]
dt = 0.005

encoder = sim_neurons.load_ppf_encoder_2D_vel_tuning(fname, dt=dt)
states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
decoding_states = ['hand_vx', 'hand_vz', 'offset'] 
beta_dec = train.inflate(beta, decoding_states, states, axis=1)
decoder = train._train_PPFDecoder_sim_known_beta(beta_dec, encoder.units, dt=dt, dist_units='m')


# initialze estimate of beta
beta_est = beta.copy()
beta_est[:,0:2] = 0

# Initialize learner and updater
n_iter = 30000.
batch_time = 60.
batch_size = batch_time/dt
half_life = 120.
rho = np.exp(np.log(0.5) / (half_life/batch_time))

learner = clda.BatchLearner(batch_size)

beta_hist = []

n_iter = X.shape[0]
spike_counts = np.zeros([n_iter, n_neurons])
decoded_output_new = np.zeros([7, n_iter])

updater = clda.PPFSmoothbatchSingleThread()
# intended_kin, spike_counts, rho, C_old, drives_neurons

spike_counts = data['spike_counts']
for n in range(1, n_iter):
    if n % 1000 == 0: print n
    #spike_counts[n-1, :] = encoder(X[n,:])
    learner(spike_counts[n-1, :].reshape(-1,1), X[n,:])
    if learner.is_full():
        # calc beta est from batch
        intended_kinematics, spike_counts_batch = learner.get_batch()
        #intended_kinematics = np.vstack([intended_kinematics, np.ones(intended_kinematics.shape[1])])
        beta_hist.append(beta_est)
        new_params = updater.calc(intended_kinematics, spike_counts_batch, rho, beta_est, drives_neurons=np.array([True, True]))
        beta_est = new_params['filt.C']
        #beta_hat, = ppfdecoder.PointProcessFilter.MLE_obs_model(intended_kinematics, neuraldata, include_offset=True)
        #beta_est = (1-rho)*beta_hat + rho*beta_est
        
beta_hist = np.dstack(beta_hist).transpose([2,0,1])


plt.figure()
axes = plot.subplots(5, 4, return_flat=True, hold=True)
for k in range(n_neurons):
    axes[k].plot(beta_hist[:,k,0])

plt.figure()
axes = plot.subplots(5, 4, return_flat=True, hold=True)
for k in range(n_neurons):
    axes[k].plot(beta_hist[:,k,1])

plt.figure()
axes = plot.subplots(2, 1, return_flat=True, hold=True)
axes[0].plot(beta[:,0])
axes[0].plot(beta_est[:,0])
axes[1].plot(beta[:,1])
axes[1].plot(beta_est[:,1])
plt.show()
