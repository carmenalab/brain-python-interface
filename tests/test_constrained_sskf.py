#!/usr/bin/python
"""
Test case for SDVKF implementation in python
"""
# TODO NEXT: one cost function with configurable return type
from scipy.io import loadmat
decoder_file = '/Users/sgowda/bmi/media/decoder/jeev072312_VFB_Kawf_B100_NS5_NU16_Z1_smoothbatch_smoothbatch_swapUnits072812_smoothbatch_rmv123a073012.mat'
decoder_data = loadmat(decoder_file)

logfile = '/Users/sgowda/example.log'
logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)
A     = np.mat(decoder_data['decoder']['A'][0,0])
W     = np.mat(decoder_data['decoder']['W'][0,0])
H     = np.mat(decoder_data['decoder']['H'][0,0])
Q_unc = np.mat(decoder_data['decoder']['Q'][0,0])

H[:,0:2] = 0

a = np.mean(np.diag(A[2:4, 2:4]))
w = np.mean(np.diag(W[2:4, 2:4]))
A[2:4, 2:4] = a * np.eye(2)
W[2:4, 2:4] = w * np.eye(2)

Q_hat_inv = np.mat(Q_unc.I)
Q_hat = Q_unc

states = ['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset']
states_to_bound = ['hand_px', 'hand_pz']
neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
drives_neurons = np.array([x in neuron_driving_states for x in states])

stochastic_states = ['hand_vx', 'hand_vz']
is_stochastic = np.array([x in stochastic_states for x in states])

# TODO extension: separate out the equality constraints and the orthogonality constraints
import riglib.bmi.kfdecoder
Q = riglib.bmi.kfdecoder.project_Q(H[:,is_stochastic], Q_hat)

# create a riglib.bmi.KalmanFilter object and compute the steady-state Kalman filter
from riglib.bmi.kfdecoder import KalmanFilter
kf = KalmanFilter(A, W, H, Q_unc)
[F, K] = kf.get_sskf()

# In one of the other arguments to the decoder updater?
H_new = H.copy()
H_new[:, ~drives_neurons] = 0
kf_new = KalmanFilter(A, W, H_new, Q)
F_new, K_new = kf_new.get_sskf()
print "\n\n\n"
print "T="
print F_new[0:2,0:2]
print "S="
print F_new[0:2,2:4]
print "N="
print F_new[2:4,2:4]
print "M="
print F_new[2:4,0:2]
