import argparse
import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import pickle

from ismore.common_state_lists import *
from utils.constants import *


# parse command line arguments
parser = argparse.ArgumentParser(description='Parse ArmAssist and/or ReHand \
    trajectories from a .hdf file (corresponding to a task for recording or \
    playing back trajectories) and save them to a .pkl file. Interpolates \
    trajectories from a "record" task, but not for a "playback" task.')
parser.add_argument('hdf_name', help='.hdf file from which to parse trajectories.')
args = parser.parse_args()

# load task, and armassist and/or rehand data from hdf file
hdf = tables.openFile(args.hdf_name)
task      = hdf.root.task
task_msgs = hdf.root.task_msgs
aa_flag = 'armassist' in hdf.root
rh_flag = 'rehand' in hdf.root
if aa_flag:
    armassist = hdf.root.armassist
if rh_flag:
    rehand = hdf.root.rehand

# determine type of task (record vs. playback)
if 'command_vel' in hdf.root.task.colnames:  # was a playback trajectories task
    INTERPOLATE_TRAJ = False
    pkl_name = 'traj_playback.pkl'
else:                                        # was a record trajectories task
    INTERPOLATE_TRAJ = True                  
    pkl_name = 'traj_reference_interp.pkl'


# code below will create a dictionary of trajectories, indexed by trial_type
traj = dict()

# idxs into task_msgs corresponding to instances when the task entered the 
# 'trial' state
trial_start_msg_idxs = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == 'trial']

# iterate over trials
for msg_idx in trial_start_msg_idxs:

    # task iteration at which this trial started
    idx_start = task_msgs[msg_idx]['time']

    trial_type = task[idx_start]['trial_type']

    # only save one trajectory for each trial type (the first one)
    if trial_type not in traj:
        print 'adding trajectory for trial type', trial_type
        
        traj[trial_type] = dict()

        # task iteration at which this trial ended
        idx_end = task_msgs[msg_idx+1]['time'] - 1

        # actual start and end times of this trial 
        ts_start = task[idx_start]['ts']  # secs
        ts_end   = task[idx_end]['ts']    # secs

        traj[trial_type]['ts_start'] = ts_start
        traj[trial_type]['ts_end']   = ts_end


        # save task data
        idxs = [idx for idx in range(len(task[:])) if idx_start <= idx <= idx_end]
        traj[trial_type]['task'] = task[idxs]

        if INTERPOLATE_TRAJ:
            # finely-spaced vector of time-stamps onto which we will interpolate armassist and rehand data
            ts_step = 0.010  # seconds (equal to 10 ms)
            ts_interp = np.arange(ts_start, ts_end, ts_step)
            df_ts_interp = pd.DataFrame(ts_interp, columns=['ts'])

        # save armassist data
        if aa_flag:
            idxs = [i for (i, ts) in enumerate(armassist[:]['ts_arrival']) if ts_start <= ts <= ts_end]
                
            if INTERPOLATE_TRAJ:
                # add one more idx to the beginning and end, if possible
                if idxs[0] != 0:
                    idxs = [idxs[0]-1] + idxs
                if idxs[-1] != len(armassist[:])-1:
                    idxs = idxs + [idxs[-1]+1]

                df_aa = df_ts_interp.copy()
                for state in aa_pos_states:
                    ts_data    = armassist[idxs]['ts_arrival']
                    state_data = armassist[idxs]['data'][state]

                    # linear interpolation
                    #interp_fn = interp1d(ts_data, state_data)
                    #interp_state_data = interp_fn(ts_interp)
                    # spline interpolation
                    from scipy.interpolate import splrep, splev
                    tck = splrep(ts_data, state_data, s=3)
                    interp_state_data = splev(ts_interp, tck)

                    df_tmp = pd.DataFrame(interp_state_data, columns=[state])
                    df_aa  = pd.concat([df_aa, df_tmp], axis=1)
    
            else:
                df_aa1 = pd.DataFrame(armassist[idxs]['data'],       columns=aa_pos_states)
                df_aa2 = pd.DataFrame(armassist[idxs]['ts_arrival'], columns=['ts'])
                df_aa  = pd.concat([df_aa1, df_aa2], axis=1)

            traj[trial_type]['armassist'] = df_aa

        # save rehand data
        if rh_flag:
            idxs = [i for (i, ts) in enumerate(rehand[:]['ts_arrival']) if ts_start <= ts <= ts_end]
                
            if INTERPOLATE_TRAJ:
                # add one more idx to the beginning and end, if possible
                if idxs[0] != 0:
                    idxs = [idxs[0]-1] + idxs
                if idxs[-1] != len(rehand[:])-1:
                    idxs = idxs + [idxs[-1]+1]

                df_rh = df_ts_interp.copy()
                for state in rh_pos_states+rh_vel_states:
                    ts_data    = rehand[idxs]['ts_arrival']
                    state_data = rehand[idxs]['data'][state]
                    interp_fn = interp1d(ts_data, state_data)
                    interp_state_data = interp_fn(ts_interp)
                    df_tmp = pd.DataFrame(interp_state_data, columns=[state])
                    df_rh  = pd.concat([df_rh, df_tmp], axis=1)
    
            else:
                df_rh1 = pd.DataFrame(rehand[idxs]['data'],       columns=rh_pos_states+rh_vel_states)
                df_rh2 = pd.DataFrame(rehand[idxs]['ts_arrival'], columns=['ts'])
                df_rh  = pd.concat([df_rh1, df_rh2], axis=1)

            traj[trial_type]['rehand'] = df_rh

        # also save armassist+rehand data into a single combined dataframe
        if INTERPOLATE_TRAJ:
            df_traj = df_ts_interp.copy()

            if aa_flag:
                for state in aa_pos_states:
                    df_traj = pd.concat([df_traj, df_aa[state]], axis=1)
            
            if rh_flag:
                for state in rh_pos_states + rh_vel_states:
                    df_traj = pd.concat([df_traj, df_rh[state]], axis=1)
            
            traj[trial_type]['traj'] = df_traj

pickle.dump(traj, open(pkl_name, 'wb'))
