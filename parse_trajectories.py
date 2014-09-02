import tables
import pickle

# hdf = tables.openFile('/storage/rawdata/hdf/test20140827_05.hdf')
# pkl_name = 'traj.pkl'
# hdf = tables.openFile('/storage/rawdata/hdf/test20140827_10.hdf')
# pkl_name = 'traj2.pkl'

hdf = tables.openFile('/storage/rawdata/hdf/test20140827_17.hdf')
pkl_name = 'traj.pkl'
hdf = tables.openFile('/storage/rawdata/hdf/test20140827_19.hdf')
pkl_name = 'traj2.pkl'

task      = hdf.root.task
task_msgs = hdf.root.task_msgs

armassist = hdf.root.armassist

trajectories = dict()

trial_idxs = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == 'trial']
wait_idxs  = [idx for (idx, msg) in enumerate(task_msgs[:]['msg']) if msg == 'wait']

for idx in trial_idxs:
    t_start = task_msgs[idx]['time']

    trial_type = task[t_start]['trial_type']
    print trial_type
    
    if not (trial_type in trajectories):
        trajectories[trial_type] = dict()

        t_end = task_msgs[idx+1]['time'] - 1
        
        ts_start = task[t_start]['ts'][0]
        ts_end   = task[t_end]['ts'][0]

        idxs = [i for (i, x) in enumerate(armassist[:]) if ts_start <= x['ts_arrival']/1e6 <= ts_end]

        # print 'ts_start:', ts_start
        # print 'first ts:', armassist[idxs[0]]['ts_arrival'] * 1e-6
        # print 'last ts:', armassist[idxs[-1]]['ts_arrival'] * 1e-6
        # print 'ts_end:', ts_end

        trajectories[trial_type]['ts_start']  = ts_start
        trajectories[trial_type]['ts_end']    = ts_end
        trajectories[trial_type]['armassist'] = armassist[idxs]



        idxs = [i for i in range(len(task[:])) if t_start <= i <= t_end]
        try:
            trajectories[trial_type]['aim_pos'] = hdf.root.task[idxs]['aim_pos']
            trajectories[trial_type]['plant_pos'] = hdf.root.task[idxs]['plant_pos']
            trajectories[trial_type]['command_vel'] = hdf.root.task[idxs]['command_vel']
        except:
            pass

pickle.dump(trajectories, open(pkl_name, 'wb'))
