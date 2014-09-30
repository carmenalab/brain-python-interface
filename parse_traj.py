import scipy.io as sio
import numpy as np
import pandas as pd
import tables
import pickle

from utils.constants import *


hdf = tables.openFile('/storage/rawdata/hdf/test20140930_02.hdf')
pkl_name = 'traj_psi.pkl'

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

        traj[trial_type]['ts_start']  = ts_start
        traj[trial_type]['ts_end']    = ts_end

        idxs = [i for (i, x) in enumerate(armassist[:]) if ts_start <= x['ts_arrival']/1e6 <= ts_end]

        df1 = pd.DataFrame(np.array(armassist[idxs]['data'].tolist()).T, index=['aa_px', 'aa_py', 'aa_ppsi'])
        aa_px_ts = (np.array(armassist[idxs]['ts'].tolist()).T)[0:1, :]
        df2 = pd.DataFrame(aa_px_ts, index=['ts'])
        df_aa = pd.concat([df1, df2])
        traj[trial_type]['armassist'] = df_aa

        if rehand_flag:
            df_rh = pd.DataFrame(np.array(rehand[idxs]['data'].tolist()), index=['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono'])
            traj[trial_type]['rehand']  = df_rh 

        idxs = [i for i in range(len(task[:])) if t_start <= i <= t_end]

        traj[trial_type]['task'] = task[idxs]
        try:
            traj[trial_type]['task'] = task[idxs]
        except:
            pass

pickle.dump(traj, open(pkl_name, 'wb'))
