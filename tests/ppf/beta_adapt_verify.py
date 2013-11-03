
from scipy.io import loadmat
import numpy as np
import math
import time
import cProfile
from riglib.bmi import train


# TODO
# 7-col version of beta
# cm decoder



data = loadmat('/Users/sgowda/Desktop/ppf_code_1023/jeev100713_VFB_PPF_B100_NS5_NU13_Z1_from1020_from1030_cont_rmv81_contData.mat')

batch_idx = 0

spike_counts = data['spike_counts'].astype(np.float64)
intended_kin = data['intended_kin']
beta_hat = data['beta_hat']
aimPos = data['aimPos']
n_iter = data['n_iter'][0,0]
stimulant_index = data['stimulant_index']
param_noise_variances = data['param_noise_variances'].ravel()
stoch_beta_index = data['stoch_beta_index']
det_beta_index = data['det_beta_index']

offset_last = 1#False
class PPFContinuousBayesianUpdater(object):
    def __init__(self, decoder, units='m'):
        self.n_units = decoder.filt.C.shape[0]
        #self.param_noise_variances = param_noise_variances
        if units == 'm':
            vel_gain = 1e-4
        else:
            vel_gain = 1e-8

        if offset_last:
            param_noise_variances = np.array([vel_gain*0.13, vel_gain*0.13,1e-4*0.06/50, ])
        else:
            param_noise_variances = np.array([1e-4*0.06/50, vel_gain*0.13, vel_gain*0.13,])
        
        self.W = np.mat(np.diag(param_noise_variances))

        self.P_params_est_old = np.zeros([self.n_units, 3, 3])
        for j in range(self.n_units):
            self.P_params_est_old[j,:,:] = self.W #Cov_params_init
        #self.P_params_est_old = P_params_est_old

        self.neuron_driving_state_inds = np.nonzero(decoder.drives_neurons)[0]
        self.neuron_driving_states = list(np.take(decoder.states, np.nonzero(decoder.drives_neurons)[0]))
        self.n_states = len(decoder.states)
        self.full_size = len(decoder.states)

        self.dt = 0.005
        if offset_last:
            self.beta_est = np.array(decoder.filt.C) #[:,self.neuron_driving_state_inds])
        else:
            self.beta_est = beta_hat[:,:,0].T

    def __call__(self, spike_obs, int_kin, rho, decoder):
        beta_est = self.beta_est[:,self.neuron_driving_state_inds]
        P_params_est_old = self.P_params_est_old
        #dt = self.dt
        #self.beta_est, self.P_params_est_old = PPF_adaptive_beta(
        #    spike_obs, int_kin, self.beta_est, self.P_params_est_old, self.dt)

        if offset_last:
            int_kin = np.hstack([int_kin, 1])
        else:
            int_kin = np.hstack([1, int_kin])
        Loglambda_predict = np.dot(int_kin, beta_est.T)#beta_hat[:,:,idx])
        rates = np.exp(Loglambda_predict)
        if np.any(rates > 1):
            print 'stuff'
            rates[rates > 1] = 1
        #lambda_predict = np.exp(Loglambda_predict)/dt
        #rates = lambda_predict*dt
        unpred_spikes = spike_obs - rates

        C_xpose_C = np.mat(np.outer(int_kin, int_kin))

        P_params_est = np.zeros([self.n_units, 3, 3])
        #beta_est_new = np.zeros([n_units, 3])
        for c in range(self.n_units):
            P_pred = P_params_est_old[c] + self.W
            P_params_est[c] = (P_pred.I + rates[c]*C_xpose_C).I
            #beta_est_new[c] = beta_est[:,c] + np.dot(int_kin, np.asarray(P_params_est[c]))*unpred_spikes[c]#

        beta_est += (unpred_spikes * np.dot(int_kin, P_params_est).T).T

        # inflate beta_est and store
        self.beta_est = np.zeros([self.n_units, self.n_states])
        self.beta_est[:,self.neuron_driving_state_inds] = beta_est

        #self.beta_est = beta_est_new
        self.P_params_est_old = P_params_est
        #return , P_params_est 

        return self.beta_est, self.P_params_est_old


## Create the object representing the initial decoder
init_beta = beta_hat[:,:,0]
init_beta = np.vstack([init_beta[1:,:], init_beta[0,:]]).T
decoder = train._train_PPFDecoder_sim_known_beta(init_beta, units=[], dist_units='m')

updater = PPFContinuousBayesianUpdater(decoder, units='m')

dt = 0.005
beta_hat_recon_error = np.nan * np.ones(beta_hat.shape)
inds = []
n_iter = 2000
for idx in range(1, n_iter):
    if idx % 1000 == 0: 
        try:
            print idx, np.max(np.abs(beta_hat_recon_error[:,:,inds]))
        except:
            pass
    if not np.any(np.isnan(aimPos[:, idx])):
        ##[test, P_params_est_old] = PPF_adaptive_beta(
        ##    spike_counts[:, idx].astype(np.float64),
        ##    intended_kin[2:4, batch_idx],
        ##    beta_hat[:,:,idx], P_params_est_old, 
        ##    param_noise_variances.ravel(), dt)

        [test, W_params_est_old] = updater(
            spike_counts[:, idx], intended_kin[2:4, batch_idx], -1, decoder)

        ## manipulate 'test' into MATLB format
        test = test[:,updater.neuron_driving_state_inds]
        #test[:,0:2] *= 100 # convert from cm to m
        test = test.T
        test = np.vstack([test[-1], test[0:2]])

        beta_hat_recon_error[:,:,idx+1] = beta_hat[:,:,idx+1] - test
        inds.append(idx+1)
        batch_idx += 1

inds = np.array(inds)
print np.max(np.abs(beta_hat_recon_error[:,:,inds]))














y = spike_counts[:,idx].astype(np.float64)
int_kin = intended_kin[2:4,batch_idx]
current_beta = beta_hat[:,:,idx]
def fn():
    for k in range(100):
        updater(y, int_kin)

#cProfile.run('fn()')
def PPF_adaptive_beta(spike_obs, int_kin, beta_est, P_params_est_old, dt):
    '''docs'''
    
    n_units = beta_est.shape[1]
    int_kin = np.hstack([1, int_kin])
    Loglambda_predict = np.dot(int_kin, beta_hat[:,:,idx])
    lambda_predict = np.exp(Loglambda_predict)/dt
    rates = lambda_predict*dt
    unpred_spikes = spike_obs - rates

    C_xpose_C = np.mat(np.outer(int_kin, int_kin))

    P_params_est = np.zeros([n_units, 3, 3])
    beta_est_new = np.zeros([n_units, 3])
    for c in range(n_units):
        P_pred = P_params_est_old[c] + W
        P_params_est[c] = (P_pred.I + rates[c]*C_xpose_C).I
        beta_est_new[c] = beta_est[:,c] + np.dot(int_kin, np.asarray(P_params_est[c]))*unpred_spikes[c]#

    return beta_est_new.T, P_params_est 


##C = data['C'][0,0]
##Cov_params_init = data['Cov_params_init']
##P_params_est_old = np.zeros([C, 3, 3])
##for j in range(C):
##    P_params_est_old[j,:,:] = Cov_params_init


