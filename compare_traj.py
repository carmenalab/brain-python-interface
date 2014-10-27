import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import *


def norm_vec(x, eps=1e-9):
    return x / (np.linalg.norm(x) + eps)


trial_type = 'touch 0'

# traj_file1 = 'traj_reference.pkl'
traj_file1 = 'traj_reference_interp.pkl'
traj_file2 = 'traj_playback.pkl'

plot_closest_idx_lines = False
plot_xy_aim_lines      = False
plot_psi_aim_lines     = False
armassist              = True
rehand                 = False #True


aa_pos_states = ['aa_px', 'aa_py', 'aa_ppsi']
rh_pos_states = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
rh_vel_states = ['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']


traj1 = pickle.load(open(traj_file1, 'rb'))
traj2 = pickle.load(open(traj_file2, 'rb'))

if armassist:
    aa1 = traj1[trial_type]['armassist']
    aa2 = traj2[trial_type]['armassist']
    aa1_len_t = us_to_s * (aa1.ix['ts', aa1.columns[-1]] - aa1.ix['ts', 0])
    aa2_len_t = us_to_s * (aa2.ix['ts', aa2.columns[-1]] - aa2.ix['ts', 0])
    print "length of aa1:", aa1.shape[1]
    print "length of aa2:", aa2.shape[1]
    print "length of aa1 (secs):", aa1_len_t
    print "length of aa2 (secs):", aa2_len_t

if rehand:
    rh1 = traj1[trial_type]['rehand']
    rh2 = traj2[trial_type]['rehand']
    rh1_len_t = us_to_s * (rh1.ix['ts', rh1.columns[-1]] - rh1.ix['ts', 0])
    rh2_len_t = us_to_s * (rh2.ix['ts', rh2.columns[-1]] - rh2.ix['ts', 0])
    print "length of rh1:", rh1.shape[1]
    print "length of rh2:", rh2.shape[1]
    print "length of rh1 (secs):", rh1_len_t
    print "length of rh2 (secs):", rh2_len_t


aim_pos     = traj2[trial_type]['task']['aim_pos']
idx_aim     = traj2[trial_type]['task']['idx_aim']
idx_aim_psi = traj2[trial_type]['task']['idx_aim_psi']
plant_pos   = traj2[trial_type]['task']['plant_pos']
command_vel = traj2[trial_type]['task']['command_vel']

idx_traj = traj2[trial_type]['task']['idx_traj']


############
# PLOTTING #
############

# # plot command velocities
# fig = plt.figure()
# plt.title('command velocities')
# grid = (4, 1)
# for i in range(2):
#     ax = plt.subplot2grid(grid, (i, 0))
#     plt.plot(command_vel[:, i], color='blue')
# ax = plt.subplot2grid(grid, (2, 0))
# plt.plot(rad_to_deg * command_vel[:, 2], color='blue')
# command_xy_speed = np.array([np.sqrt(np.sum(vel[0:2]**2)) for vel in command_vel])
# ax = plt.subplot2grid(grid, (3, 0))
# plt.plot(command_xy_speed, color='blue')


if armassist:

    fig = plt.figure()
    plt.title('xy trajectories')

    plt.plot(aa1.ix['aa_px', :], aa1.ix['aa_py', :], '-D', color='red', markersize=5)
    plt.plot(aa2.ix['aa_px', :], aa2.ix['aa_py', :], '-D', color='blue', markersize=2.5)

    if plot_closest_idx_lines:
        # plot lines to indicate points closest to ref trajectory
        for (idx, p_pos, c_vel) in zip(idx_traj, plant_pos, command_vel):
            plt.plot([p_pos[0], aa1.ix['aa_px', idx]], [p_pos[1], aa1.ix['aa_py', idx]], color='gray')

    if plot_xy_aim_lines:
        # plot lines to indicate xy aiming position
        for (a_pos, p_pos, c_vel) in zip(aim_pos, plant_pos, command_vel):
            plt.plot([p_pos[0], a_pos[0]], [p_pos[1], a_pos[1]], color='gray')

    if plot_psi_aim_lines:
        # plot lines to indicate psi aiming position
        for (idx, p_pos, c_vel) in zip(idx_aim_psi, plant_pos, command_vel):
            plt.plot([p_pos[0], aa1.ix['aa_px', idx]], [p_pos[1], aa1.ix['aa_py', idx]], color='gray')

    # plot first plant position in green
    plt.plot(aa1.ix['aa_px', 0], aa1.ix['aa_py', 0], 'D', color='green', markersize=10)  


    fig = plt.figure()
    for idx, (plant_psi, aim_psi, cmd_vel_psi) in enumerate(zip(plant_pos[:,2], aim_pos[:,2], command_vel[:,2])):
        plt.plot([idx], [rad_to_deg*aa1.ix['aa_ppsi', idx_traj[idx]]], 'D', color='gray', markersize=2.5)
        plt.plot([idx], [rad_to_deg*plant_psi], 'D', color='blue', markersize=2.5)
        plt.plot([idx], [rad_to_deg*aim_psi],   'D', color='green', markersize=2.5)
        plt.plot([idx, idx], [rad_to_deg*plant_psi, rad_to_deg*aim_psi], color='black', markersize=2.5)
        plt.plot([idx, idx],  \
                 [rad_to_deg*plant_psi, rad_to_deg*(plant_psi + cmd_vel_psi)],  \
                 color='red', markersize=2.5)

    fig = plt.figure()
    plt.title('ArmAssist trajectories (time-warped)')
    grid = (3, 1)

    for i, state in enumerate(aa_pos_states):
        ax = plt.subplot2grid(grid, (i, 0))
        if state == 'aa_ppsi':
            scale = rad_to_deg    
        else:
            scale = 1
        plt.plot(scale * aa1.ix[state, idx_traj.reshape(-1)], color='red')
        plt.plot(scale * plant_pos[:, i], 'D', color='blue', markersize=2.5)


    delta_pos = np.diff(aa1.ix[['aa_px', 'aa_py'], :])
    delta_psi = rad_to_deg * np.diff(aa1.ix['aa_ppsi', :])
    delta_ts  = us_to_s * np.diff(aa1.ix['ts', :])
    xy_vel_traj1  = np.array([np.sqrt(np.sum(x**2)) for x in delta_pos.T]) / delta_ts
    psi_vel_traj1 = delta_psi / delta_ts
    traj1_tvec = np.array(us_to_s * aa1.ix['ts', :])
    traj1_tvec -= traj1_tvec[0]

    fig = plt.figure()
    grid = (3, 1)
    ax = plt.subplot2grid(grid, (0, 0))
    plt.plot(traj1_tvec[:-1], xy_vel_traj1, color='red')
    ax = plt.subplot2grid(grid, (1, 0))
    plt.plot(traj1_tvec[:-1], psi_vel_traj1, color='red')
    ax = plt.subplot2grid(grid, (2, 0))
    plt.plot(traj1_tvec, rad_to_deg * aa1.ix['aa_ppsi', :], color='red')


    delta_pos = np.diff(aa2.ix[['aa_px', 'aa_py'], :])
    delta_psi = rad_to_deg * np.diff(aa2.ix['aa_ppsi', :])
    delta_ts  = us_to_s * np.diff(aa2.ix['ts', :])
    xy_vel_traj2  = np.array([np.sqrt(np.sum(x**2)) for x in delta_pos.T]) / delta_ts
    psi_vel_traj2 = delta_psi / delta_ts
    traj2_tvec = np.array(us_to_s * aa2.ix['ts', :])
    traj2_tvec -= traj2_tvec[0]

    fig = plt.figure()
    grid = (3, 1)
    ax = plt.subplot2grid(grid, (0, 0))
    plt.plot(traj2_tvec[:-1], xy_vel_traj2, color='blue')
    ax = plt.subplot2grid(grid, (1, 0))
    plt.plot(traj2_tvec[:-1], psi_vel_traj2, color='blue')
    ax = plt.subplot2grid(grid, (2, 0))
    plt.plot(traj2_tvec, rad_to_deg * aa2.ix['aa_ppsi', :], color='blue')


# fig = plt.figure()
# plt.title('psi trajectories')
# plt.plot(rad_to_deg * aa1.ix['aa_ppsi', :], color='red')
# plt.plot(rad_to_deg * aa2.ix['aa_ppsi', :], color='blue')

# fig = plt.figure()
# plt.title('ReHand trajectories (not time-warped)')
# grid = (4, 1)
# for i, state in enumerate(rh_pos_states):
#     ax = plt.subplot2grid(grid, (i, 0))
#     plt.plot(rad_to_deg * rh1.ix[state, :], color='red')
#     plt.plot(rad_to_deg * plant_pos[:, 3+i], color='blue')


if rehand:

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
