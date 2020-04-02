'''Test script for ArmAssist and ArmAssistPIController.'''

import time
import numpy as np
from math import sin, cos
import armassist

import matplotlib.pyplot as plt

# constants
pi = np.pi
L1 = armassist.L1
L2 = armassist.L2

cm_to_m = 0.01
m_to_cm = 100.

real_time_plot = True

ftime = 20  # final time of simulation - you can change it
tstep = 0.1  # decoder tstep
t_vec = np.arange(0, ftime, tstep)

# for data saving
data_bf         = np.zeros((3, len(t_vec)))
data_wf         = np.zeros((3, len(t_vec)))
data_bf_dot     = np.zeros((3, len(t_vec)))
data_wf_dot     = np.zeros((3, len(t_vec)))
data_wf_dot_ref = np.zeros((3, len(t_vec)))
data_cmc_pos    = np.zeros((2, len(t_vec)))

# create and start ArmAssist object (includes ArmAssist and its PIC)
aa_tstep = 0.005
aa_pic_tstep = 0.01
KP = np.mat([[-10.,   0.,  0.],
             [  0., -20.,  0.],
             [  0.,   0., 20.]])  # P gain matrix
TI = 0.1*np.identity(3)  # I gain matrix

aa = armassist.ArmAssist(aa_tstep, aa_pic_tstep, KP, TI)
aa.daemon = True
aa.start()

# get initial state of ArmAssist
state  = aa.get_state()
bf     = state['bf']
wf     = state['wf']
bf_dot = state['bf_dot']
wf_dot = state['wf_dot']

t_sim_start = time.time()
for i, t in enumerate(t_vec):
    t_itr_start = time.time()

    # desired values in the global frame
    des_x_vel     = 5           # cm/s
    des_y_vel     = 5 * sin(t)  # cm/s
    des_z_ang_vel = 0           # rad/s
    wf_dot_ref = np.mat([des_x_vel, des_y_vel, des_z_ang_vel]).T

    # aa_pic.update_reference(wf_dot_ref)
    aa.update_reference(wf_dot_ref)


    state = aa.get_state()
    
    # convert from cm and cm/s to m and m/s
    state['bf'][0:2] *= cm_to_m
    state['wf'][0:2] *= cm_to_m
    state['bf_dot'][0:2] *= cm_to_m
    state['wf_dot'][0:2] *= cm_to_m

    bf = state['bf']
    wf = state['wf']
    bf_dot = state['bf_dot']
    wf_dot = state['wf_dot']

    # data saving
    data_bf[:, i]         = bf.reshape(3)
    data_wf[:, i]         = wf.reshape(3)
    data_bf_dot[:, i]     = bf_dot.reshape(3)
    data_wf_dot[:, i]     = wf_dot.reshape(3)
    data_wf_dot_ref[:, i] = wf_dot_ref.reshape(3)

    # real-time plotting
    if real_time_plot:
        x   = float(wf[0])
        y   = float(wf[1])
        psi = float(wf[2])
        wh1_x = x - L1*sin(psi)
        wh1_y = y + L1*cos(psi)
        wh2_x = x + L2*sin(32.5/180*pi + psi)
        wh2_y = y - L2*cos(32.5/180*pi + psi)
        wh3_x = x - L2*sin(32.5/180*pi - psi)
        wh3_y = y - L2*cos(32.5/180*pi - psi)

        data_cmc_pos[0, i] = x
        data_cmc_pos[1, i] = y

        if i == 0:
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.hold(True)

            h_tri, = plt.plot([wh1_x, wh2_x, wh3_x, wh1_x], [wh1_y, wh2_y, wh3_y, wh1_y])
            h_cmc, = plt.plot([x], [y], 'ro')
            h_traj, = plt.plot(data_cmc_pos[0, :i+1], data_cmc_pos[1, :i+1])
            ax.axis([-1, 2, -1, 1])
            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)

            plt.ion()
            plt.show()
            plt.title('ArmAssist simulation')
            plt.xlabel('x position (m)')
            plt.ylabel('y position (m)')
        else:
            h_tri.set_xdata([wh1_x, wh2_x, wh3_x, wh1_x])
            h_tri.set_ydata([wh1_y, wh2_y, wh3_y, wh1_y])
            h_cmc.set_xdata([x])
            h_cmc.set_ydata([y])
            h_traj.set_xdata(data_cmc_pos[0, :i+1])
            h_traj.set_ydata(data_cmc_pos[1, :i+1])
            plt.draw()
    
    # if elapsed iteration time was less than tstep, then sleep
    t_elapsed = time.time() - t_itr_start
    if tstep - t_elapsed > 0:
        time.sleep(tstep - t_elapsed)
    else:
        print('Warning: iteration time is greater than tstep')

print('Total simulation time:', time.time() - t_sim_start)

# stop ArmAssist process
aa.stop()


# plot desired vs. actual velocities
titles = ['x direction', 'y direction', 'rotation about z axis']
ylabels = ['m/s', 'm/s', 'rad/s']

plt.ioff()
plt.figure(2)
for i in range(3):
    ax = plt.subplot(3, 1, i+1)
    plt.plot(t_vec, data_wf_dot_ref[i, :], t_vec, data_wf_dot[i, :])
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.axis([x_min, 1.5*x_max, y_min, y_max])
    plt.title(titles[i])
    plt.xlabel('Time (s)')
    plt.ylabel(ylabels[i])
    plt.legend(['desired', 'real'])  #, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.show(block=True)
