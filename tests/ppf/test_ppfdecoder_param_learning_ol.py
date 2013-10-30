#!/usr/bin/python
'''
Test trying to learn the beta parameters using SB and the continuous relearning
methods
'''
import numpy as np
from scipy.io import loadmat, savemat
from riglib.bmi import sim_neurons
import matplotlib.pyplot as plt
from riglib.bmi import ppfdecoder, train, clda, bmi
#import plot
import time
import cProfile
import cmath

reload(ppfdecoder)
reload(sim_neurons)
reload(train)
reload(clda)
plt.close('all')

N = 168510.
fname = 'sample_spikes_and_kinematics_%d.mat' % N 
data = loadmat(fname)
# truedata = loadmat('/Users/sgowda/bmi/workspace/adaptive_ppf/ppf_test_case_matlab_output.mat')
X = data['hand_vel'].T

beta = data['beta']
beta = np.vstack([beta[1:, :], beta[0,:]]).T
n_neurons = beta.shape[0]
dt = 0.005 #truedata['T_loop'][0,0]

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

class PPFContinuousBayesianUpdater(object):
    def __init__(self, decoder):
        n_neurons = decoder.filt.C.shape[0]
        self.n_neurons = n_neurons
        self.rho = -1

        neuron_driving_state_inds = np.nonzero(decoder.drives_neurons)[0]
        self.neuron_driving_states = list(np.take(decoder.states, np.nonzero(decoder.drives_neurons)[0]))
        n_states = len(neuron_driving_state_inds)

        self.meta_ppf = [None]*self.n_neurons
        C_init = np.array(decoder.filt.C.copy())
        for k in range(self.n_neurons):
            I = np.mat(np.eye(n_states))
            R_diag_neuron = 1e-4 * np.array([0.13, 0.13, 0.06/50])
            R = np.diag(R_diag_neuron)
            self.meta_ppf[k] = ppfdecoder.PointProcessFilter(I, R, np.zeros(n_states), dt=decoder.filt.dt)

            # Initialize meta-PPF
            init_beta_est = C_init[k, neuron_driving_state_inds]
            self.meta_ppf[k]._init_state(init_state=init_beta_est, init_cov=R)
        
    def calc(self, intended_kin, spike_counts, rho, decoder):
        if np.ndim(intended_kin) == 1:
            intended_kin = intended_kin.reshape(-1,1)
            spike_counts = spike_counts.reshape(-1,1)

        n_neurons, n_obs = spike_counts.shape
        intended_kin = np.array(intended_kin)
        C = np.mat(intended_kin[decoder.drives_neurons]).reshape(1,-1)
        #C = np.mat(intended_kin[decoder.drives_neurons]).reshape(1,-1)
        #C = np.mat(intended_kin[decoder.drives_neurons]).reshape(1,-1)
        #C_xpose_C = np.outer(C, C)
        C_xpose_C = C.T * C
        
        A = self.meta_ppf[0].A
        A_xpose = A.T
        W = self.meta_ppf[0].W
        dt = self.meta_ppf[0].dt
        for n in range(n_neurons):
            for k in range(n_obs):
                #self.meta_ppf[n](spike_counts[n,k])
                #self = self.meta_ppf[n]
                obs_t = spike_counts[n,k]
                #target_state = None
                st = self.meta_ppf[n].state

                #obs_t = np.mat(obs_t.reshape(-1,1))
                n_obs, n_states = C.shape
                
                pred_state_mean = A*st.mean
                #pred_obs = self.meta_ppf[n]._obs_prob(pred_state)

                Loglambda_predict = C * pred_state_mean
                pred_obs = cmath.exp(Loglambda_predict[0,0])/dt
                #pred_obs = np.exp(Loglambda_predict[0,0])/dt
        
                #P_pred = pred_state.cov
                P_pred = A*st.cov*A_xpose + W
                #nS = self.meta_ppf[n].A.shape[0]
        
                #q = 1./
                P_est = P_pred - (pred_obs*dt) * P_pred* C_xpose_C *P_pred
                ##if n_obs > n_states:
                ##    Q_inv = np.mat(np.diag(np.array(pred_obs).ravel() * self.meta_ppf[n].dt))
                ##    I = np.mat(np.eye(nS))
                ##    D = C.T * Q_inv * C
                ##    F = (D - D*P_pred*(I + D).I * D)
                ##    P_est = P_pred - P_pred * F * P_pred
                ##elif n_obs == 1:
                ##else:
                ##    Q_diag = (np.array(pred_obs).ravel() * self.meta_ppf[n].dt)**-1
                ##    Q = np.mat(np.diag(Q_diag))
        
                ##    P_est = P_pred - P_pred*C.T * (Q + C*P_pred*C.T).I * C*P_pred
        
                unpred_spikes = obs_t - pred_obs*self.meta_ppf[n].dt
                x_est = pred_state_mean + P_est*C.T*unpred_spikes
                post_state = bmi.GaussianState(x_est, P_est)
                self.meta_ppf[n].state = post_state


        beta_new = np.hstack([x.state.mean for x in self.meta_ppf]).T
        beta_new = train.inflate(beta_new, self.neuron_driving_states, decoder.states, axis=1)

        return {'filt.C':beta_new}



updater_cont = PPFContinuousBayesianUpdater(decoder_rml)
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

n_iter = 60
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

    st = time.time()
    new_params_rml = updater_cont.calc(int_kin, spike_counts[n-1,:], -1, decoder_rml)
    print "Calc time: %g" % (time.time() - st)
    decoder_rml.update_params(new_params_rml)

def perf_fn():
    for k in range(1000):
        updater_cont.calc(int_kin, spike_counts[n-1,:], -1, decoder_rml)

cProfile.run('perf_fn()')


if 0:
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
    
