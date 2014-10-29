import scipy.io as sio
import numpy as np
import pandas as pd
import tables
import pickle
from scipy.interpolate import interp1d


from utils.constants import *

#INTERPOLATE_TRAJ = True  # use when parsing a reference trajectory
INTERPOLATE_TRAJ = False  # use when parsing a playback trajectory

#hdf_name = '/storage/rawdata/hdf/test20141028_08.hdf'  # ref
#hdf_name = '/storage/rawdata/hdf/test20141028_09.hdf'  # playback

#hdf_name = '/storage/rawdata/hdf/test20141028_11.hdf'  # ref
hdf_name = '/storage/rawdata/hdf/test20141028_19.hdf'  # playback

if INTERPOLATE_TRAJ:
    pkl_name = 'traj_reference_interp.pkl'
else:
    pkl_name = 'traj_playback.pkl'


aa_pos_states = ['aa_px', 'aa_py', 'aa_ppsi']
rh_pos_states = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
rh_vel_states = ['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']


# load task, armassist, and rehand data from hdf file
hdf = tables.openFile(hdf_name)
task      = hdf.root.task
task_msgs = hdf.root.task_msgs

try:
    armassist = hdf.root.armassist
except:
    print 'No armassist data saved in hdf file.'
    armassist_flag = False
else:
    armassist_flag = True

try:
    rehand = hdf.root.rehand
except:
    print 'No rehand data saved in hdf file.'
    rehand_flag = False
else:
    rehand_flag = True

# create a dictionary of trajectories, indexed by trial_type
traj = dict()

trial_idxs = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == 'trial']
# for idx in trial_idxs:
for idx in trial_idxs[0:4]:  # ONLY LOOK AT FIRST 4 TRIALS FOR NOW
    t_start = task_msgs[idx]['time']
    trial_type = task[t_start]['trial_type']

    # if we haven't already saved a trajectory for this trial_type
    if trial_type not in traj:
        print 'adding trial type', trial_type
        traj[trial_type] = dict()

        t_end    = task_msgs[idx+1]['time'] - 1
        ts_start = task[t_start]['ts']
        ts_end   = task[t_end]['ts']
        traj[trial_type]['ts_start'] = ts_start
        traj[trial_type]['ts_end']   = ts_end

        # save task data
        idxs = [i for i in range(len(task[:])) if t_start <= i <= t_end]
        traj[trial_type]['task'] = task[idxs]

        if INTERPOLATE_TRAJ:
            # finely-spaced vector of time-stamps onto which we will interpolate armassist and rehand data
            ts_step = 0.010  # seconds (equal to 10 ms)
            ts_vec = np.arange(ts_start, ts_end, ts_step)

        # save armassist data
        if armassist_flag:
            if INTERPOLATE_TRAJ:
                idxs = [i for (i, x) in enumerate(armassist[:]) if ts_start <= x['ts_arrival'] <= ts_end]
                if idxs[0] != 0:
                    idxs = [idxs[0]-1] + idxs
                if idxs[-1] != len(armassist[:])-1:
                    idxs = idxs + [idxs[-1]+1]

                # TODO -- make same fixes to rehand code below
                # for each traj, position data and arrival_ts is being saved, 
                #   but arrival_ts is being saved with label 'ts' -- need to fix this / make this more clear 

                # what to name the column?
                df_aa = pd.DataFrame(ts_vec, columns=['ts']).T
                for state in aa_pos_states:
                    ts_data    = armassist[idxs]['ts_arrival']
                    state_data = armassist[idxs]['data'][state]
                    interp_fn = interp1d(ts_data, state_data)
                    df_tmp = pd.DataFrame(interp_fn(ts_vec), columns=[state]).T
                    df_aa  = pd.concat([df_aa, df_tmp])
                traj[trial_type]['armassist'] = df_aa
            else:
                idxs = [i for (i, x) in enumerate(armassist[:]) if ts_start <= x['ts_arrival'] <= ts_end]
                df_aa1 = pd.DataFrame(np.array(armassist[idxs]['data'].tolist()).T, index=aa_pos_states)
                aa_ts = armassist[idxs]['ts']  # TODO -- use ts_arrival, not ts
                df_aa2 = pd.DataFrame(aa_ts, index=['ts'])
                traj[trial_type]['armassist'] = pd.concat([df_aa1, df_aa2])

        # save rehand data
        if rehand_flag:
            if INTERPOLATE_TRAJ:
                idxs = [i for (i, x) in enumerate(rehand[:]) if ts_start <= x['ts_arrival'] <= ts_end]
                if idxs[0] != 0:
                    idxs = [idxs[0]-1] + idxs
                if idxs[-1] != len(rehand[:])-1:
                    idxs = idxs + [idxs[-1]+1]

                df_rh = pd.DataFrame(ts_vec, columns=['ts']).T
                for state in rh_pos_states+rh_vel_states:
                    ts_data    = rehand[idxs]['ts_arrival'][state]
                    state_data = rehand[idxs]['data'][state]
                    interp_fn = interp1d(ts_data, state_data)
                    df_tmp = pd.DataFrame(interp_fn(ts_vec), columns=[state]).T
                    df_rh  = pd.concat([df_rh, df_tmp])

                traj[trial_type]['rehand'] = df_rh
            else:
                idxs = [i for (i, x) in enumerate(rehand[:]) if ts_start <= x['ts_arrival'] <= ts_end]
                df_rh1 = pd.DataFrame(np.array(rehand[idxs]['data'].tolist()).T, 
                                      index=rh_pos_states+rh_vel_states)
                rh_ts = rehand[idxs]['ts']  # TODO -- use ts_arrival, not ts
                df_rh2 = pd.DataFrame(rh_ts, index=['ts'])
                traj[trial_type]['rehand'] = pd.concat([df_rh1, df_rh2])


        if INTERPOLATE_TRAJ:
            # also form a single dataframe named traj
            # TODO -- this is probably very inefficient, there should be a way to merge df_aa and df_rh
            df_final = pd.DataFrame(ts_vec, columns=['ts']).T
            if armassist_flag:
                for state in aa_pos_states:
                    df_tmp = pd.DataFrame(df_aa.T[state], columns=[state]).T
                    df_final = pd.concat([df_final, df_tmp])
            if rehand_flag:
                for state in rh_pos_states + rh_vel_states:
                    df_tmp = pd.DataFrame(df_rh.T[state], columns=[state]).T
                    df_final = pd.concat([df_final, df_rh[state]])
            traj[trial_type]['traj'] = df_final

pickle.dump(traj, open(pkl_name, 'wb'))
