#!/usr/bin/python
'''
Test case for PPFDecoder
'''
import numpy as np
from scipy.io import loadmat, savemat
import utils
from riglib.bmi import sim_neurons

reload(sim_neurons)

kin_data = loadmat('paco_hand_kin.mat')
hand_kin = kin_data['hand_kin']
hand_vel = hand_kin[2:4, :]
X = utils.mat.pad_ones(hand_vel, axis=0, pad_ind=0).T
X2 = utils.mat.pad_ones(hand_vel, axis=0, pad_ind=-1).T

C = 20
Delta = 0.005
baseline_hz = 10
baseline = np.log(baseline_hz)
max_speed = 0.3 # m/s
max_rate = 70 # hz
mod_depth = (np.log(max_rate)-np.log(baseline_hz))/max_speed

pref_angle_data = loadmat('preferred_angle_c50.mat')
pref_angles = pref_angle_data['preferred_angle'].ravel()
pref_angles = pref_angles[:C]

# Load MATLAB sim results for comparison
N = 168510
N = 10000
data = loadmat('sample_spikes_and_kinematics_%d.mat' % N)
spike_counts = data['spike_counts']
beta = data['beta']

X = utils.mat.pad_ones(data['hand_vel'], axis=0, pad_ind=0).T

dt = 0.005

N = data['hand_vel'].shape[1]
k = 0
spikes = np.zeros([N, C])

## for k in range(C):
##     tau_samples = data['tau_samples'][0][k].ravel().tolist()
##     point_proc = sim_neurons.PointProcess(beta[:,k], dt, tau_samples=tau_samples)
##     spikes[:,k] = point_proc.sim_batch(X[0:N, :]) 
## 
##     matching = np.array_equal(spike_counts[:,k], spikes[:,k])
##     print k, matching

init_state = X[0,:]
tau_samples = [data['tau_samples'][0][k].ravel().tolist() for k in range(C)]
ensemble = sim_neurons.PointProcessEnsemble(beta, init_state, dt, tau_samples=tau_samples)

spikes_ensemble = np.zeros([N, C])
for n in range(1, N):
    spikes_ensemble[n-1, :] = ensemble(X[n,:])

print np.array_equal(spikes_ensemble, spike_counts)
