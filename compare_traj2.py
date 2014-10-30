import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import *


def norm_vec(x, eps=1e-9):
    return x / (np.linalg.norm(x) + eps)


trial_type = 'touch 0'

traj_file_ref = 'traj_reference_interp.pkl'
traj_file_pbk = 'traj_playback.pkl'

plot_closest_idx_lines = False
plot_xy_aim_lines      = True
plot_psi_aim_lines     = False

aa_pos_states = ['aa_px', 'aa_py', 'aa_ppsi']
rh_pos_states = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
rh_vel_states = ['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']


traj_ref = pickle.load(open(traj_file_ref, 'rb'))
traj_pbk = pickle.load(open(traj_file_pbk, 'rb'))

aa_flag = 'armassist' in traj_pbk[trial_type]
rh_flag = 'rehand' in traj_pbk[trial_type]


if aa_flag:
    aa_ref = traj_ref[trial_type]['armassist'].T
    aa_pbk = traj_pbk[trial_type]['armassist'].T

    print "length of aa_ref:", len(aa_ref)
    print "length of aa_pbk:", len(aa_pbk)
    print "length of aa_ref (secs):", aa_ref['ts'][aa_ref.index[-1]] - aa_ref['ts'][0]
    print "length of aa_pbk (secs):", aa_pbk['ts'][aa_pbk.index[-1]] - aa_pbk['ts'][0]

if rh_flag:
    rh_ref = traj_ref[trial_type]['rehand'].T
    rh_pbk = traj_pbk[trial_type]['rehand'].T

    print "length of rh_ref:", len(rh_ref)
    print "length of rh_pbk:", len(rh_pbk)
    print "length of rh_ref (secs):", rh_ref['ts'][rh_ref.index[-1]] - rh_ref['ts'][0]
    print "length of rh_pbk (secs):", rh_pbk['ts'][rh_pbk.index[-1]] - rh_pbk['ts'][0]

task_pbk       = traj_pbk[trial_type]['task']
aim_pos        = task_pbk['aim_pos']
idx_aim        = task_pbk['idx_aim']
idx_aim_psi    = task_pbk['idx_aim_psi']
plant_pos      = task_pbk['plant_pos']
command_vel    = task_pbk['command_vel']
idx_traj       = task_pbk['idx_traj']

############
# PLOTTING #
############

task_tvec = task_pbk['ts'] - task_pbk['ts'][0]

color_ref = 'red'
color_pbk = 'blue'

tight_layout_kwargs = {
    'pad':   0.5,
    'w_pad': 0.5,
    'h_pad': 0.5,
}

if aa_flag:
    aa_tvec = aa_pbk['ts'] - aa_pbk['ts'][0]

    fig = plt.figure()
    plt.title('xy trajectories')

    plt.plot(aa_ref['aa_px'], aa_ref['aa_py'], '-D', color=color_ref,  markersize=5)
    plt.plot(aa_pbk['aa_px'], aa_pbk['aa_py'], '-D', color=color_pbk, markersize=2.5)

    # lines to indicate points closest to ref trajectory
    if plot_closest_idx_lines:
        for (idx, p_pos, c_vel) in zip(idx_traj, plant_pos, command_vel):
            plt.plot([p_pos[0], aa_ref['aa_px'][idx]], 
                     [p_pos[1], aa_ref['aa_py'][idx]], 
                     color='gray')

    # lines to indicate xy aiming position
    if plot_xy_aim_lines:
        for (a_pos, p_pos, c_vel) in zip(aim_pos, plant_pos, command_vel):
            plt.plot([p_pos[0], a_pos[0]], 
                     [p_pos[1], a_pos[1]], 
                     color='green')

    # lines to indicate psi aiming position
    if plot_psi_aim_lines:
        for (idx, p_pos, c_vel) in zip(idx_aim_psi, plant_pos, command_vel):
            plt.plot([p_pos[0], aa_ref['aa_px'][idx]], 
                     [p_pos[1], aa_ref['aa_py'][idx]], 
                     color='black')

    # plot first plant position in green
    plt.plot(aa_ref['aa_px'][0], aa_ref['aa_py'][0], 'D', color='green', markersize=10)

    plt.tight_layout(**tight_layout_kwargs)


    fig = plt.figure()
    plt.title('psi playback analysis')
    
    time_warped_ref_psi_traj = np.array([aa_ref['aa_ppsi'][idx_traj[idx]] for idx in range(len(idx_traj))])
    plt.plot(task_tvec, rad_to_deg * time_warped_ref_psi_traj, '-D', color=color_ref, markersize=2.5)
    plt.plot(task_tvec, rad_to_deg * plant_pos[:,2],           '-D', color=color_pbk, markersize=2.5)

    for idx, (plant_psi, aim_psi, cmd_vel_psi) in enumerate(zip(plant_pos[:,2], aim_pos[:,2], command_vel[:,2])):
        # line showing which psi value the playback algorithm was aiming towards
        plt.plot([task_tvec[idx], task_tvec[idx]], [rad_to_deg*plant_psi, rad_to_deg*aim_psi], color='green', markersize=2.5)
        ## line showing command velocity
        #plt.plot([task_tvec[idx], task_tvec[idx]], [rad_to_deg*plant_psi, rad_to_deg*(plant_psi + cmd_vel_psi)], color='gray',  markersize=2.5)


    # compare playback trajectory vs. time-warped reference trajectory
    fig = plt.figure()
    grid = (3, 1)

    for i, state in enumerate(aa_pos_states):
        ax = plt.subplot2grid(grid, (i, 0))
        ax.set_title(state + ' trajectories (time-warped)')
        if state == 'aa_ppsi':
            scale = rad_to_deg    
        else:
            scale = 1
        plt.plot(task_tvec, scale * aa_ref[state][idx_traj.reshape(-1)], color=color_ref)
        plt.plot(task_tvec, scale * plant_pos[:, i], 'D', color=color_pbk, markersize=2.5)

    plt.tight_layout(**tight_layout_kwargs)

    # calculate and plot xy and psi velocity during reference and playback trajectories
    for aa, color in zip([aa_ref, aa_pbk], [color_ref, color_pbk]):
        fig = plt.figure()
        grid = (2, 1)

        delta_pos = np.diff(aa[['aa_px', 'aa_py']], axis=0)
        delta_psi = rad_to_deg * np.diff(aa['aa_ppsi'])
        delta_ts  = np.diff(aa['ts'])
        xy_vel  = np.array([np.sqrt(np.sum(x**2)) for x in delta_pos]) / delta_ts
        psi_vel = delta_psi / delta_ts
        tvec = aa['ts'] - aa['ts'][0]

        ax = plt.subplot2grid(grid, (0, 0))
        ax.set_title('xy velocity')
        plt.plot(tvec[:-1], xy_vel, color=color)
        ax = plt.subplot2grid(grid, (1, 0))
        ax.set_title('psi velocity')
        plt.plot(tvec[:-1], psi_vel, color=color)

        plt.tight_layout(**tight_layout_kwargs)

    # plot playback command velocities as saved by task
    fig = plt.figure()
    grid = (4, 1)
    
    ax = plt.subplot2grid(grid, (0, 0))
    ax.set_title('command xy speed')
    command_xy_speed = np.array([np.sqrt(np.sum(vel[0:2]**2)) for vel in command_vel])
    plt.plot(command_xy_speed, color='blue')
    
    for i, state in enumerate(aa_pos_states):
        ax = plt.subplot2grid(grid, (i+1, 0))
        ax.set_title(state + ' command velocity')
        if state == 'aa_ppsi':
            scale = rad_to_deg    
        else:
            scale = 1
        plt.plot(task_tvec, scale * command_vel[:, i], color=color_pbk)


if rh_flag:
    rh_tvec = rh_pbk['ts'] - rh_pbk['ts'][0]

    if aa_flag:
        offset = 3
    else:
        offset = 0

    # compare playback trajectory vs. time-warped reference trajectory
    fig = plt.figure()
    grid = (4, 1)

    for i, state in enumerate(rh_pos_states):
        ax = plt.subplot2grid(grid, (i, 0))
        ax.set_title(state + ' trajectories (time-warped)')
        plt.plot(task_tvec, rad_to_deg * rh_ref[state][idx_traj.reshape(-1)], color=color_ref)
        plt.plot(task_tvec, rad_to_deg * plant_pos[:, i+offset], color=color_pbk)

    plt.tight_layout(**tight_layout_kwargs)

plt.show()
