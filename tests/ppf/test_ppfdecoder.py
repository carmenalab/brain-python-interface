#!/usr/bin/python
'''
Test case for PPFDecoder
'''
import numpy as np
from scipy.io import loadmat, savemat
import utils
from riglib.bmi import sim_neurons
import matplotlib.pyplot as plt
from riglib.bmi import ppfdecoder, state_space_models as ssm
from scipy.io import loadmat, savemat
from riglib.bmi.sim_neurons import PointProcessEnsemble
import matplotlib.pyplot as plt
from riglib.bmi import state_space_models

reload(ppfdecoder)
reload(sim_neurons)
plt.close('all')

N = 10000
data = loadmat('sample_spikes_and_kinematics_%d.mat' % N)
truedata = loadmat('/Users/sgowda/bmi/workspace/adaptive_ppf/ppf_test_case_matlab_output.mat')
beta = data['beta']
hand_vel = data['hand_vel']

beta = np.vstack([beta[1:, :], beta[0,:]])
C = beta.shape[1]

X = data['hand_vel'].T
#X = utils.mat.pad_ones(data['hand_vel'], axis=0, pad_ind=-1).T
dt = 0.005
N = data['hand_vel'].shape[1]
n_iter = N

init_state = np.hstack([X[0,:], 1])
tau_samples = [data['tau_samples'][0][k].ravel().tolist() for k in range(C)]
ensemble = sim_neurons.PointProcessEnsemble(beta, init_state, dt, tau_samples=tau_samples)

spike_counts = np.zeros([N, C])
for n in range(1, N):
    spike_counts[n-1, :] = ensemble(X[n,:])

print "Python sim spikes matches MATLAB's: %s" % np.array_equal(spike_counts, data['spike_counts'])

def inflate(A, current_states, full_state_ls):
    nS = len(full_state_ls)#A.shape[0]
    A_new = np.zeros([nS, A.shape[1]])
    new_inds = [full_state_ls.index(x) for x in current_states]
    A_new[new_inds, :] = A
    return A_new

states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
decoding_states = ['hand_vx', 'hand_vz', 'offset'] 

dt = 0.005
A, W = state_space_models.linear_kinarm_kf(update_rate=dt, units_mult=1)
n_neurons = beta.shape[1]
beta = inflate(beta, decoding_states, states)

#W = inflate(W, decoding_states, states)
ppf = ppfdecoder.PointProcessFilter(A, W, beta.T, dt)
ppf._init_state()

# Fake units for PPFDecoder
units = np.vstack([(x, 1) for x in range(n_neurons)])
bounding_box = (np.array([]), np.array([]), )
drives_neurons = ['hand_vx', 'hand_vz', 'offset']
states_to_bound = []
dec = ppfdecoder.PPFDecoder(ppf, units, bounding_box, states, drives_neurons,
        states_to_bound)

decoded_output_new = np.zeros([7, n_iter])
for idx in range(0, n_iter/3):
    sl = slice(3*idx+1,3*(idx+1)+1)
    decoded_output_new[:,sl] = dec(spike_counts[sl].T)
    
x_est = truedata['x_est']
print np.max(np.abs(x_est[0,:n_iter:dec.n_subbins] - decoded_output_new[3,::dec.n_subbins]))
print np.max(np.abs(x_est[0,:n_iter] - decoded_output_new[3,:]))

plt.figure()
plt.hold(True)
plt.plot(np.arange(0, n_iter, dec.n_subbins), decoded_output_new[3,:n_iter:dec.n_subbins], label='pyth')
plt.plot(x_est[0,:n_iter], label='matlab')
plt.plot(hand_vel[0,:n_iter], label='handvel')
plt.legend()
plt.show()
