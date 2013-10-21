#!/usr/bin/python
'''
Test case for PPFDecoder

TODO sim ensemble needs to be wrapped in something that will return timestamps
'''
import numpy as np
from scipy.io import loadmat, savemat
from riglib.bmi import sim_neurons
import matplotlib.pyplot as plt
from riglib.bmi import ppfdecoder, state_space_models, train
from scipy.io import loadmat, savemat

reload(ppfdecoder)
reload(sim_neurons)
reload(train)
plt.close('all')

N = 10000
data = loadmat('sample_spikes_and_kinematics_%d.mat' % N)
truedata = loadmat('/Users/sgowda/bmi/workspace/adaptive_ppf/ppf_test_case_matlab_output.mat')
#hand_vel = data['hand_vel']

beta = data['beta']
beta = np.vstack([beta[1:, :], beta[0,:]])

X = data['hand_vel'].T

n_iter = data['hand_vel'].shape[1]
n_neurons = beta.shape[1]
dt = 0.005

init_state = np.hstack([X[0,:], 1])
tau_samples = [data['tau_samples'][0][k].ravel().tolist() for k in range(n_neurons)]
ensemble = sim_neurons.PointProcessEnsemble(beta.T, init_state, dt, tau_samples=tau_samples)

states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
decoding_states = ['hand_vx', 'hand_vz', 'offset'] 

beta = train.inflate(beta, decoding_states, states)

dec = train._train_PPFDecoder_sim_known_beta(beta, ensemble.units, dt=dt, dist_units='m')

spike_counts = np.zeros([n_iter, n_neurons])
decoded_output_new = np.zeros([7, n_iter])
for n in range(1, n_iter):
    spike_counts[n-1, :] = ensemble(X[n,:])
    decoded_output_new[:, n-1] = dec.predict(spike_counts[n-1 ,:])

## for idx in range(0, n_iter/3):
##     sl = slice(3*idx+1,3*(idx+1)+1)
##     decoded_output_new[:,sl] = dec(spike_counts[sl].T)
    
print "Python sim spikes matches MATLAB's: %s" % np.array_equal(spike_counts, data['spike_counts'])

x_est = truedata['x_est']
print np.max(np.abs(x_est[0,:n_iter:dec.n_subbins] - decoded_output_new[3,::dec.n_subbins]))
print np.max(np.abs(x_est[0,:n_iter-1] - decoded_output_new[3,:-1]))

plt.figure()
plt.hold(True)
plt.plot(decoded_output_new[3,:n_iter], label='pyth')
#plt.plot(np.arange(0, n_iter, dec.n_subbins), decoded_output_new[3,:n_iter:dec.n_subbins], label='pyth')
plt.plot(x_est[0,:n_iter], label='matlab')
plt.plot(X[:n_iter, 0], label='handvel')
plt.legend()
plt.show()
