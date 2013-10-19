
#!/usr/bin/python
"""
Test to compare PPF loop calculations against Maryam's MATLAB code
"""
import numpy as np
from riglib.bmi import ppfdecoder, state_space_models as ssm
from scipy.io import loadmat, savemat
from riglib.bmi.sim_neurons import PointProcessEnsemble
import matplotlib.pyplot as plt

reload(ppfdecoder)

data = loadmat('sample_spikes_and_kinematics_10000.mat')
hand_vel = data['hand_vel']
beta = data['beta']
beta = np.vstack([beta[1:, :], beta[0,:]])
spike_counts = data['spike_counts']
hand_vel = data['hand_vel']

n_iter = 10000
T_loop = 0.005

Delta_KF = 0.1
a_kf = 0.8
w_kf = 0.0007
A_kf = np.diag([a_kf, a_kf, 1])
W_kf = np.diag([w_kf, w_kf, 0])
A, W = ssm.resample_ssm(A_kf, W_kf, Delta_old=Delta_KF, Delta_new=T_loop)

a_ppf = 9.889048329050316e-01
w_ppf = 4.290850 * 1e-05;

# Instantiate the PPF
##ppf = ppfdecoder.PointProcessFilter(A, W, beta, T_loop)
##ppf._init_state()

##point_proc = PointProcessEnsemble(beta, T_loop)

##decoded_output = np.zeros([3, n_iter])
##for idx in range(1, n_iter):
##    # TODO generate spike counts from point process simulator, not file!
##    y_t = point_proc(hand_vel[:,idx])
##    ppf(y_t)
##    decoded_output[:,idx] = ppf.get_mean()
##
##plt.figure()
##plt.hold(True)
##plt.plot(decoded_output[0,:])
##plt.plot(hand_vel[0,:n_iter])
##plt.show()

# Compare against MATLAB data
truedata = loadmat('/Users/sgowda/bmi/workspace/adaptive_ppf/ppf_test_case_matlab_output.mat')
a_ppf = truedata['A'][0,0]
w_ppf = truedata['W'][0,0]
A = np.mat(np.diag([a_ppf, a_ppf, 1]))
W = np.mat(np.diag([w_ppf, w_ppf, 0]))
ppf = ppfdecoder.PointProcessFilter(A, W, beta, T_loop)
ppf._init_state()

decoded_output = np.zeros([3, n_iter])
for idx in range(1, n_iter):
    ppf(spike_counts[idx, :])
    decoded_output[:,idx] = ppf.get_mean()
    
x_est = truedata['x_est']
plt.figure()
plt.hold(True)
plt.plot(x_est[0,:n_iter], label='matlab')
plt.plot(hand_vel[0,:n_iter], label='handvel')
plt.plot(decoded_output[0,:], label='pyth')
plt.legend()
plt.show()


print np.max(np.abs(x_est[0,:n_iter] - decoded_output[0,:]))


# TODO expand A, W, C to same dimensions as for KF
truedata = loadmat('/Users/sgowda/bmi/workspace/adaptive_ppf/ppf_test_case_matlab_output.mat')
a_ppf = truedata['A'][0,0]
w_ppf = truedata['W'][0,0]
A = np.mat(np.diag([a_ppf, a_ppf, 1]))
W = np.mat(np.diag([w_ppf, w_ppf, 0]))
ppf = ppfdecoder.PointProcessFilter(A, W, beta, T_loop)
ppf._init_state()

decoded_output = np.zeros([3, n_iter])
for idx in range(1, n_iter):
    ppf(spike_counts[idx, :])
    decoded_output[:,idx] = ppf.get_mean()
    
x_est = truedata['x_est']
plt.figure()
plt.hold(True)
plt.plot(x_est[0,:n_iter], label='matlab')
plt.plot(hand_vel[0,:n_iter], label='handvel')
plt.plot(decoded_output[0,:], label='pyth')
plt.legend()
plt.show()


print np.max(np.abs(x_est[0,:n_iter] - decoded_output[0,:]))


