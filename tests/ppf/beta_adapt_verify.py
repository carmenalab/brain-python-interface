
from scipy.io import loadmat
import numpy as np
import math
import time
import cProfile

data = loadmat('/Users/sgowda/Desktop/ppf_code_1023/jeev100713_VFB_PPF_B100_NS5_NU13_Z1_from1020_from1030_cont_rmv81_contData.mat')
batch_idx = 0

spike_counts = data['spike_counts']
intended_kin = data['intended_kin']
beta_hat = data['beta_hat']
aimPos = data['aimPos']
n_iter = data['n_iter'][0,0]
stimulant_index = data['stimulant_index']
param_noise_variances = data['param_noise_variances'].ravel()
stoch_beta_index = data['stoch_beta_index']
det_beta_index = data['det_beta_index']

def PPF_adaptive_beta(spike_obs, int_kin, beta_est, P_params_est_old, 
        param_noise_variances, dt):
    '''docs'''
    
    def _update(lambda_predict, beta_est, P_params_prev, obs):
        #beta_est = np.mat(beta_est).reshape(-1,1)
        P_pred = P_params_prev + W
        #Loglambda_predict = int_kin * beta_est #beta_est.T * int_kin
        #lambda_predict = np.min([np.real(math.exp(Loglambda_predict[0,0])), 1])/dt
        P_params_est = (P_pred.I + (lambda_predict*dt)*C_xpose_C).I
        beta_est = beta_est + np.dot(int_kin, np.asarray(P_params_est))*(obs - lambda_predict*dt)
        #import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        return beta_est, P_params_est

    n_units = beta_est.shape[1]
    int_kin = np.hstack([1, int_kin])
    Loglambda_predict = np.dot(int_kin, beta_hat[:,:,idx])
    lambda_predict = np.exp(Loglambda_predict)/dt
    rates = lambda_predict*dt
    unpred_spikes = spike_obs - rates

    C_xpose_C = np.mat(np.outer(int_kin, int_kin))

    #int_kin = np.vstack([1, int_kin.reshape(-1,1)])
    #int_kin = np.mat(int_kin).reshape(1,-1)
    P_params_est = np.zeros([n_units, 3, 3])
    #beta_est_new = np.mat(np.zeros([3, n_units]))
    beta_est_new = np.zeros([n_units, 3])
    for c in range(n_units):
        #beta_est_new[:,c], P_params_est[:,:,c] = _update(lambda_predict[c], beta_est[:,c], P_params_est_old[:,:,c], spike_obs[c])
        P_pred = P_params_est_old[c] + W
        P_params_est[c] = (P_pred.I + rates[c]*C_xpose_C).I
        beta_est_new[c] = beta_est[:,c] + np.dot(int_kin, np.asarray(P_params_est[c]))*unpred_spikes[c]#

    return beta_est_new.T, P_params_est 

C = data['C'][0,0]
Cov_params_init = data['Cov_params_init']
W_params_est_old = np.zeros([C, 3, 3])
for j in range(C):
    W_params_est_old[j,:,:] = Cov_params_init

W = np.mat(np.diag(param_noise_variances))
## idx = 64
## st = time.time()
## for k in range(100):
## 
## print (time.time() - st)/10
## print beta_next - beta_hat[:,:,idx+1]

dt = 0.005
beta_hat_recon_error = np.nan * np.ones(beta_hat.shape)
inds = []
n_iter = 1000
for idx in range(1, n_iter):
    if not np.any(np.isnan(aimPos[:, idx])):
        [test, W_params_est_old] = PPF_adaptive_beta(
            spike_counts[:, idx].astype(np.float64),
            intended_kin[2:4, batch_idx],
            beta_hat[:,:,idx], W_params_est_old, 
            param_noise_variances.ravel(), dt)

        beta_hat_recon_error[:,:,idx+1] = beta_hat[:,:,idx+1] - test
        inds.append(idx+1)
        #if np.any(beta_hat_recon_error[:,:,idx+1]):
        #    print 'error'
        batch_idx += 1

inds = np.array(inds)
print np.max(np.abs(beta_hat_recon_error[:,:,inds]))

y = spike_counts[:,idx].astype(np.float64)
int_kin = intended_kin[2:4,batch_idx]
current_beta = beta_hat[:,:,idx]
def fn():
    for k in range(100):
        st = time.time()
        PPF_adaptive_beta(y, int_kin, current_beta, W_params_est_old, param_noise_variances, 0.005)
        print time.time() - st

cProfile.run('fn()')        
