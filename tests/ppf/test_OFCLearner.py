
from scipy.io import loadmat
import numpy as np
from riglib.bmi import clda


data = loadmat('/Users/sgowda/Desktop/ppf_code_1023/jeev100713_VFB_PPF_B100_NS5_NU13_Z1_from1020_from1030_cont_rmv81_contData.mat')

T_loop = data['T_loop'][0,0]
cursor_kin = data['cursor_kin']
aimPos = data['aimPos']
n_iter = data['n_iter'][0,0]
intended_kin = data['intended_kin']

# Make AimPOS 3D
aimPos = np.vstack([aimPos[0,:], np.zeros(n_iter), aimPos[1,:]])

learner = clda.OFCLearner3DEndptPPF(n_iter, dt=T_loop)

kin_state = np.zeros([7, cursor_kin.shape[1]])
kin_state[-1,:] = 1
state_inds_2d = [0,2,3,5]
kin_state[state_inds_2d, :] = cursor_kin

for idx in range(1, n_iter):
    if not np.any(np.isnan(aimPos[:, idx])):
        learner(np.mat([[0]]), kin_state[:,idx], aimPos[:,idx], -1, 'target')

kin_data, _ = learner.get_batch()
learner_error = intended_kin[:,:kin_data.shape[1]] - np.array(kin_data[state_inds_2d,:])
print(np.max(np.abs(learner_error)))
