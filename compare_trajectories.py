import pickle
import numpy as np
import matplotlib.pyplot as plt

rad_to_deg = 180 / np.pi

colors = dict()
colors['red']   = (1, 0, 0)
colors['green'] = (0, 1, 0)
colors['blue']  = (0, 0, 1)

trial_type = 'touch red'
# trial_type = 'pinch grip green'

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

aim_pos_traj = trajectories2[trial_type]['aim_pos']
print "length of aim_pos_traj:", len(aim_pos_traj)

plant_pos_traj = trajectories2[trial_type]['plant_pos']
print "length of plant_pos_traj:", len(plant_pos_traj)

command_vel_traj = trajectories2[trial_type]['command_vel']



print "length of traj1 (secs):", traj1_len_t
print "length of traj2 (secs):", traj2_len_t


fig = plt.figure()
for idx, pos in enumerate(traj1[:]['data']):
    plt.plot([pos[0]], [pos[1]], 'D', color=colors['red'])
# for idx, pos in enumerate(traj2[:]['data']):
#     plt.plot([pos[0]], [pos[1]], 'D', color=colors['blue'])
x = [pos[0] for pos in traj2[:]['data']]
y = [pos[1] for pos in traj2[:]['data']]
plt.plot(x, y, color=colors['blue'])


fig2 = plt.figure()
for idx, pos in enumerate(traj1[:]['data']):
    plt.plot([idx], [pos[2]*rad_to_deg], 'D', color=colors['red'])
for idx, pos in enumerate(traj2[:]['data']):
    plt.plot([idx], [pos[2]*rad_to_deg], 'D', color=colors['blue'])


fig3 = plt.figure()
for idx, (a_pos, p_pos, c_vel) in enumerate(zip(aim_pos_traj, plant_pos_traj, command_vel_traj)):
    plt.plot([a_pos[0]], [a_pos[1]], 'D', color=colors['red'])
    if np.sum(np.abs(c_vel)) < 1e-9:
        c = colors['green']
    else:
        c = colors['blue']
    plt.plot([p_pos[0]], [p_pos[1]], 'D', color=c)
    plt.plot([a_pos[0], p_pos[0]], [a_pos[1], p_pos[1]])



for idx, (a_pos, p_pos, c_vel) in enumerate(zip(aim_pos_traj, plant_pos_traj, command_vel_traj)):
    print 'aim_pos', a_pos
    print 'plant_pos', p_pos
    print 'command_vel', c_vel
    print ''


plt.show()


