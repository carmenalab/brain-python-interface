'''Test script for ReHand.'''

import time
import numpy as np
from math import sin, cos
import rehand

import matplotlib.pyplot as plt

# constants
pi = np.pi

real_time_plot = True
len_thumb = 1
len_index = 1.5
len_fing3 = 2
len_prono = 2.5

ftime = 20  # final time of simulation - you can change it
tstep = 0.1  # decoder tstep
t_vec = np.arange(0, ftime, tstep)

# for data saving
data_pos = np.zeros((4, len(t_vec)))
data_vel = np.zeros((4, len(t_vec)))

# create and start ReHand object
rh = rehand.ReHand(tstep=0.005)
rh.daemon = True
rh.start()


t_sim_start = time.time()
for i, t in enumerate(t_vec):
    t_itr_start = time.time()

    # desired values in the global frame
    des_thumb_vel = 1           # rad/s
    des_index_vel = 1 * sin(t)  # rad/s
    des_fing3_vel = 0           # rad/s
    des_prono_vel = 1 * cos(t)  # rad/s
    des_vel = np.mat([des_thumb_vel, des_index_vel, des_fing3_vel, des_prono_vel]).T

    rh.set_vel(des_vel)

    state = rh.get_state()

    pos = state['pos']
    vel = state['vel']
    
    # data saving
    data_pos[:, i] = pos.reshape(4)
    data_vel[:, i] = vel.reshape(4)

    # real-time plotting
    if real_time_plot:
        # angular positions
        p_thumb = float(pos[0])
        p_index = float(pos[1])
        p_fing3 = float(pos[2])
        p_prono = float(pos[3])

        if i == 0:
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.hold(True)

            h_thumb, = plt.plot([0, len_thumb*cos(p_thumb)], [0, len_thumb*sin(p_thumb)])
            h_index, = plt.plot([0, len_index*cos(p_index)], [0, len_index*sin(p_index)])
            h_fing3, = plt.plot([0, len_fing3*cos(p_fing3)], [0, len_fing3*sin(p_fing3)])
            h_prono, = plt.plot([0, len_prono*cos(p_prono)], [0, len_prono*sin(p_prono)])

            ax.axis([-3, 3, -3, 3])
            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)

            plt.ion()
            plt.show()
            plt.title('ReHand simulation')
        else:
            h_thumb.set_xdata([0, len_thumb*cos(p_thumb)])
            h_thumb.set_ydata([0, len_thumb*sin(p_thumb)])
            h_index.set_xdata([0, len_index*cos(p_index)])
            h_index.set_ydata([0, len_index*sin(p_index)])
            h_fing3.set_xdata([0, len_fing3*cos(p_fing3)])
            h_fing3.set_ydata([0, len_fing3*sin(p_fing3)])
            h_prono.set_xdata([0, len_prono*cos(p_prono)])
            h_prono.set_ydata([0, len_prono*sin(p_prono)])
            plt.draw()
    
    # if elapsed iteration time was less than tstep, then sleep
    t_elapsed = time.time() - t_itr_start
    if tstep - t_elapsed > 0:
        time.sleep(tstep - t_elapsed)
    else:
        print('Warning: iteration time is greater than tstep')

print('Total simulation time:', time.time() - t_sim_start)

# stop ReHand process
rh.stop()
