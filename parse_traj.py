import scipy.io as sio
import numpy as np
import pandas as pd
import tables
import pickle

from utils.constants import *


hdf_name = '/storage/rawdata/hdf/test20140930_02.hdf'
pkl_name = 'traj_playback.pkl'


aa_pos_states = ['aa_px', 'aa_py', 'aa_ppsi']
rh_pos_states = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']

# load task, armassist, and rehand data from hdf file
hdf = tables.openFile(hdf_name)
task      = hdf.root.task
task_msgs = hdf.root.task_msgs
armassist = hdf.root.armassist
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
for idx in trial_idxs:
    t_start = task_msgs[idx]['time']
    trial_type = task[t_start]['trial_type']

    # if we haven't already saved a trajectory for this trial_type
    if trial_type not in traj:
        traj[trial_type] = dict()

        t_end    = task_msgs[idx+1]['time'] - 1
        ts_start = task[t_start]['ts'][0]
        ts_end   = task[t_end]['ts'][0]
        traj[trial_type]['ts_start'] = ts_start
        traj[trial_type]['ts_end']   = ts_end

        # save task data
        idxs = [i for i in range(len(task[:])) if t_start <= i <= t_end]
        traj[trial_type]['task'] = task[idxs]

        # save armassist data
        idxs = [i for (i, x) in enumerate(armassist[:]) if ts_start <= us_to_s*x['ts_arrival'] <= ts_end]
        df_aa1 = pd.DataFrame(np.array(armassist[idxs]['data'].tolist()).T, index=aa_pos_states)
        aa_px_ts = (np.array(armassist[idxs]['ts'].tolist()).T)[0:1, :]
        df_aa2 = pd.DataFrame(aa_px_ts, index=['ts'])
        traj[trial_type]['armassist'] = pd.concat([df_aa1, df_aa2])

        # save rehand data
        if rehand_flag:
            idxs = [i for (i, x) in enumerate(rehand[:]) if ts_start <= us_to_s*x['ts_arrival'] <= ts_end]
            traj[trial_type]['rehand'] = pd.DataFrame(np.array(rehand[idxs]['data'].tolist()).T, index=rh_pos_states)

pickle.dump(traj, open(pkl_name, 'wb'))
