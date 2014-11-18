import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import *


def norm_vec(x, eps=1e-9):
    return x / (np.linalg.norm(x) + eps)


trial_type = 'touch 0'

traj_file1 = 'traj_reference_interp.pkl'

plot_closest_idx_lines = False
plot_xy_aim_lines      = False
plot_psi_aim_lines     = False
armassist              = True
rehand                 = False #True


aa_pos_states = ['aa_px', 'aa_py', 'aa_ppsi']
rh_pos_states = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
rh_vel_states = ['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']


traj1 = pickle.load(open(traj_file1, 'rb'))

if armassist:
    aa1 = traj1[trial_type]['armassist']
    aa1_len_t = aa1.ix['ts', aa1.columns[-1]] - aa1.ix['ts', 0]
    print "length of aa1:", aa1.shape[1]
    print "length of aa1 (secs):", aa1_len_t

if rehand:
    rh1 = traj1[trial_type]['rehand']
    rh1_len_t = rh1.ix['ts', rh1.columns[-1]] - rh1.ix['ts', 0]
    print "length of rh1:", rh1.shape[1]
    print "length of rh1 (secs):", rh1_len_t



############
# PLOTTING #
############


if armassist:

    fig = plt.figure()
    plt.title('xy trajectories')
    plt.plot(aa1.ix['aa_px', :], aa1.ix['aa_py', :], '-D', color='red', markersize=5)
    plt.plot(aa1.ix['aa_px', 0], aa1.ix['aa_py', 0], 'D', color='green', markersize=10)  # plot first pos in green

    fig = plt.figure()
    plt.title('psi trajectories')
    plt.plot(rad_to_deg * aa1.ix['aa_ppsi', :], color='red')
    


if False: #rehand:

    fig = plt.figure()
    plt.title('ReHand trajectories (time-warped)')
    grid = (4, 1)

    if armassist:
        offset = 3
    else:
        offset = 0
    for i, state in enumerate(rh_pos_states):
        ax = plt.subplot2grid(grid, (i, 0))
        plt.plot(rad_to_deg * rh1.ix[state, idx_aim.reshape(-1)], color='red')
        plt.plot(rad_to_deg * plant_pos[:, offset+i], color='blue')


# # plot rehand angles for reference trajectory
# fig = plt.figure()
# grid = (4, 1)

# for i, state in enumerate(rh_pos_states):
#     ax = plt.subplot2grid(grid, (i, 0))
#     plt.plot(rad_to_deg * rh1.ix[state, :], color='red')




plt.show()
