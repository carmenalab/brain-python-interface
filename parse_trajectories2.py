import scipy.io as sio
import numpy as np
import pandas as pd
import tables
import pickle

from utils.constants import *


hdf = tables.openFile('/storage/rawdata/hdf/test20140929_16.hdf')  # playing back psi too
pkl_name = 'traj_psi.pkl'
# hdf = tables.openFile('/storage/rawdata/hdf/test20140929_13.hdf')  # not playing back psi
# pkl_name = 'traj_no_psi.pkl'

task      = hdf.root.task
task_msgs = hdf.root.task_msgs

armassist = hdf.root.armassist
# rehand    = hdf.root.rehand

traj = dict()

trial_idxs = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == 'trial']
wait_idxs  = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == 'wait']

for idx in trial_idxs:
    t_start = task_msgs[idx]['time']

    trial_type = task[t_start]['trial_type']
    print trial_type
    
    if not (trial_type in traj):
        traj[trial_type] = dict()

        t_end = task_msgs[idx+1]['time'] - 1
        
        ts_start = task[t_start]['ts'][0]
        ts_end   = task[t_end]['ts'][0]

        idxs = [i for (i, x) in enumerate(armassist[:]) if ts_start <= x['ts_arrival']/1e6 <= ts_end]

        df1 = pd.DataFrame(np.array(armassist[idxs]['data'].tolist()).T, index=['aa_px', 'aa_py', 'aa_ppsi'])
        aa_px_ts = (np.array(armassist[idxs]['ts'].tolist()).T)[0:1, :]
        df2 = pd.DataFrame(aa_px_ts, index=['ts'])
        df_aa = pd.concat([df1, df2])
        # df_rh = pd.DataFrame(np.array(rehand[idxs]['data'].tolist()), index=['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono'])

        traj[trial_type]['ts_start']  = ts_start
        traj[trial_type]['ts_end']    = ts_end
        traj[trial_type]['armassist'] = df_aa
        # traj[trial_type]['rehand']  = df_rh 

        idxs = [i for i in range(len(task[:])) if t_start <= i <= t_end]
        try:
            traj[trial_type]['aim_pos']     = task[idxs]['aim_pos']
            traj[trial_type]['aim_idx']     = task[idxs]['aim_idx']
            traj[trial_type]['plant_pos']   = task[idxs]['plant_pos']
            traj[trial_type]['command_vel'] = task[idxs]['command_vel']
        except:
            pass

pickle.dump(traj, open(pkl_name, 'wb'))
