import pickle
import numpy as np
import matplotlib.pyplot as plt

colors = dict()
colors['red']   = (1, 0, 0)
colors['green'] = (0, 1, 0)
colors['blue']  = (0, 0, 1)

trial_type = 'touch red'

traj_file1 = 'traj.pkl'
traj_file2 = 'traj2.pkl'

trajectories1 = pickle.load(open(traj_file1, 'rb'))
trajectories2 = pickle.load(open(traj_file2, 'rb'))

traj1 = trajectories1[trial_type]['armassist']
traj2 = trajectories2[trial_type]['armassist']

traj1_len_t = 1e-6 * (traj1[-1]['ts_arrival'] - traj1[0]['ts_arrival'])
traj2_len_t = 1e-6 * (traj2[-1]['ts_arrival'] - traj2[0]['ts_arrival'])

print "length of traj1:", len(traj1)
print "length of traj2:", len(traj2)

print "length of traj1 (secs):", traj1_len_t
print "length of traj2 (secs):", traj2_len_t


fig = plt.figure()

for idx, pos in enumerate(traj1[:]['data']):
    # if idx == 0:
    #     continue
    
    plt.plot([pos[0]], [pos[1]], 'D', color=colors['red'])

for idx, pos in enumerate(traj2[:]['data']):
    # if idx == 0:
    #     continue
    
    plt.plot([pos[0]], [pos[1]], 'D', color=colors['blue'])

plt.show()


