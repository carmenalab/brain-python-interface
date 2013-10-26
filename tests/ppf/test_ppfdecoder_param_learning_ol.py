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
decoder_sb = train._train_PPFDecoder_sim_known_beta(beta_est, encoder.units, dt=dt, dist_units='m')
decoder_rml = train._train_PPFDecoder_sim_known_beta(beta_est, encoder.units, dt=dt, dist_units='m')

# Initialize learner and updater
batch_time = 60.
batch_size = batch_time/dt
half_life = 120.
rho = np.exp(np.log(0.5) / (half_life/batch_time))

learner = clda.BatchLearner(batch_size)
updater_sb = clda.PPFSmoothbatchSingleThread()
updater_sb.rho = rho


updater_cont = clda.PPFContinuousBayesianUpdater(n_neurons, decoder_rml)
updater_cont.rho = -1

## RUN 
n_iter = X.shape[0]
spike_counts = data['spike_counts']
beta_hist = []

beta_cont_hist = np.zeros([n_iter, n_neurons, 3])
beta_cont_hist[0, :, -1] = beta[:, -1]

I = np.mat(np.eye(3*n_neurons))
R_diag_neuron = 1e-4 * np.array([0.13, 0.13, 0.06/50])
R = np.diag(np.tile(R_diag_neuron, (n_neurons,)))
meta_ppf = ppfdecoder.PointProcessFilter(I, R, np.zeros(3*n_neurons), dt=dt)
meta_ppf._init_state(init_state=beta_cont_hist[0,:,:].ravel(), init_cov=R)

for n in range(1, n_iter):
    if n % batch_size == 0: print n
    int_kin = np.hstack([np.zeros(3), X[n,0], 0, X[n,1], 1])
    beta_C = np.array([X[n,0], X[n,1], 1])

    learner(spike_counts[n-1, :].reshape(-1,1), int_kin)

    if learner.is_full():
        # calc beta est from batch
        int_kin_batch, spike_counts_batch = learner.get_batch()
        beta_hist.append(decoder_sb.filt.C)
        new_params = updater_sb.calc(int_kin_batch, spike_counts_batch, rho, decoder_sb)
        decoder_sb.update_params(new_params)

    #### ## Try to predict the beta of the first unit
    #### meta_ppf.C = np.mat(beta_C.reshape(1, -1))
    #### meta_ppf_C = np.zeros([n_neurons, 3*n_neurons])
    #### for k in range(n_neurons):
    ####     meta_ppf_C[k, 3*k:3*(k+1)] = beta_C
    #### meta_ppf.C = meta_ppf_C
    #### #obs_0 = np.mat([[spike_counts[n-1, 0]]], dtype=np.float64)
    #### obs = np.mat(spike_counts[n-1, :].reshape(-1,1))
    #### beta_cont_hist[n, :] = np.array(meta_ppf(obs)).ravel().reshape(n_neurons, -1)

    new_params_rml = updater_cont.calc(int_kin, spike_counts[n-1,:], -1, decoder_rml)
    decoder_rml.update_params(new_params_rml)

        
plt.figure()
axes = plot.subplots(3, 1, return_flat=True, hold=True)
axes[0].plot(beta_cont_hist[:,0,0])
axes[1].plot(beta_cont_hist[:,0,1])
axes[2].plot(beta_cont_hist[:,0,2])

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
axes[0].plot(beta[:,0], label="True")
axes[0].plot(decoder_sb.filt.C[:,3], label='SB est')
#axes[0].plot(beta_cont_hist[-1,:,0])
axes[0].plot(decoder_rml.filt.C[:,3], label='RML est')
axes[1].plot(beta[:,1], label="True")
axes[1].plot(decoder_sb.filt.C[:,5], label="SB est")
#axes[1].plot(beta_cont_hist[-1,:,1])
axes[1].plot(decoder_rml.filt.C[:,5], label="RML est")
axes[0].legend()
plt.show()

