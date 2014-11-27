'''Direct translation of Je's Matlab ArmAssist simulation into Python.'''

from armassist import *

import numpy as np
import matplotlib.pyplot as plt

from math import sin, cos

import time

pi = np.pi

# Sample program for simulation of ArmAssist without\with external force

tstep = 0.1 # sampling frequency of control loop - you can change it
ftime = 10 # final time of simulation - you can change it

# mechanical constants of robot - don't change values
m = 3.05 # m - Robot mass (kg)
Iz = 0.067 # Iz - Robot moment inertia (kgm2) ! this is not actual value
R = 0.0254 # R - Wheel radius (m)
L1 = 0.130 # L1 - Distance from the center mouse to front wheel.
L2 = 0.153 # L2 - Distance from the center mouse to rear right (left) wheel.
n = 19 # n - Gear ratio
J0 = 0.003 # combined inertia of the motor, gear train and wheel referred to the motor shaft ! this is not acutal value
b0 = 0 # viscous-friction coefficient of the motor, gear and wheel combination

# matrices used in the model - don't change them. If you need to change, please let me know.
H = np.mat([[1./m, 0, 0], [0, 1./m, 0], [0, 0, 1./Iz]]) # [1/m 0 0;0 1/m 0;0 0 1/Iz];
B = np.mat([[-1, cos(pi/4), cos(pi/4)], [0, -sin(pi/4), sin(pi/4)], [L1, L2*cos(pi*12/180), L2*cos(pi*12/180)]]) # [-1 cos(pi/4) cos(pi/4);0 -np.sin(pi/4) np.sin(pi/4);L1 L2*cos(pi*12/180) L2*cos(pi*12/180)]
G = np.identity(3) + (H * B * B.T * n**2 * J0 / R**2) # (eye(3)+H*B*B'*n^2*J0/R^2)


# bf - vector including 3 variables of dynamic model of ArmAssist in the body frame
# bf(1) --> u - velocity component in x direction in the body frame (m/s) 
# bf(2) --> v - velocity component in y direction in the body frame (m/s)
# bf(3) --> r - anglular rate of body rototation (rad/s) 
bf = np.mat(np.zeros((3, 1))) # [0 0 0]' # initial vector of bf

# wf - vector including 3 variables of ArmAssist in the world frame
# wf(1) --> x - x position in the world frame (m) 
# wf(2) --> y - y position in the world frame (m)
# wf(3) --> psi - robot orientation angle (rad)
wf = np.mat(np.zeros((3, 1))) # [0 0 0]' # initial vector of wf

ie_bf = np.mat(np.zeros((3, 1)))  # [0;0;0] # initial values for Integral control

#start simulation

t_vec = np.arange(0, ftime, tstep)


u = np.zeros(t_vec.shape) # velocity and angular rate components in the body frame
v = np.zeros(t_vec.shape)
r = np.zeros(t_vec.shape)
x = np.zeros(t_vec.shape) # position and orientation components in the world frame
y = np.zeros(t_vec.shape)
psi = np.zeros(t_vec.shape)

des_x_v = np.zeros(t_vec.shape) # shaped refernce velocity
des_y_v = np.zeros(t_vec.shape)
des_psi_v = np.zeros(t_vec.shape)

x_v = np.zeros(t_vec.shape) # velocity and angular velocity components in the world frame
y_v = np.zeros(t_vec.shape)
psi_v = np.zeros(t_vec.shape)


t_start_ = time.time()
for i, t in enumerate(t_vec):
 
    #external force vector. Caution-external force vector should be defined in the body frame.
    ex_f_xr = 0# external force in the x-direction at the body frame (XR in the figure in the description)
    ex_f_yr = 0# external force in the y-direction at the body frame (YR in the figure in the description)
    ex_t_zr = 0# Caution: external torque in the z-direction at the body frame (ZR in the figure in the description)
    ex_f = np.mat([ex_f_xr, ex_f_yr, ex_t_zr]).T # [ex_f_xr;ex_f_yr;ex_t_zr] 

    #desired values in the global frame
    des_x_vel = 0.05# m/s
    des_y_vel = 0.05*np.sin(t)# m/s
    des_z_ang_vel = 0 #rad/s 

    #Transform desired vector in the global frame to that in the body frame
    # K_M=[cos(wf(3)) np.sin(wf(3)) 0;-np.sin(wf(3)) cos(wf(3)) 0;0 0 1]*([des_x_vel;des_y_vel;des_z_ang_vel])
    wf_2 = wf.item(2)
    K_M = np.mat([[cos(wf_2), sin(wf_2), 0], [-sin(wf_2), cos(wf_2), 0], [0, 0, 1]]) * np.mat([des_x_vel, des_y_vel, des_z_ang_vel]).T

    #design of input - you can design those values as you need.
    # PI control
    # e_bf = np.mat([K_M[0], K_M[1], K_M[2]]).T - bf # [K_M(1);K_M(2);K_M(3)] - bf # error between desired and actual
    # e_bf = np.mat([K_M.item(0), K_M.item(1), K_M.item(2)]).T - bf # [K_M(1);K_M(2);K_M(3)] - bf # error between desired and actual
    e_bf = K_M - bf # [K_M(1);K_M(2);K_M(3)] - bf # error between desired and actual
    
    ie_bf = ie_bf + e_bf*tstep # integration of error

    KP = np.mat([[-10., 0., 0.], [0., -20., 0.], [0., 0., 20.]]) # [-10. 0. 0.;0. -20. 0.;0. 0. 20.] # P gain matrix
    TI = 0.1*np.identity(3) # [0.1 0. 0.;0. 0.1 0.;0. 0. 0.1] # I gain matrix

    # if i == 0:
    #     print 'e_bf:', e_bf.T
    #     print 'ie_bf:', ie_bf.T

    torq = KP*(e_bf+TI*ie_bf) # PI Control torq(1)-motor 1 torque, torq(2)-motor 2 torque, torq(3)-motor 3 torque
    # print torq.T

    ### mobile robot dynamics ########################################

    t_start = time.time()
    n_steps = 50
    for ii in range(n_steps): # the for-structure is utilized in order to make the simulation continuous
        #intergration in body frame
        # if True:  #i == 0 and ii == 0:
        #     print 'torq:', torq.T
            # print 'bf:', bf.T
            # print 'G:', G
            # print 'H:', H
            # print 'B:', B
            # print 'R:', R
            # print 'b0:', b0
            # print 'n:', n
            # print 'ex_f:', ex_f
            # print 'lll:', tstep/n_steps

        bf, bf_dot = rungekutta_body_frame(torq, bf, G, H, B, R, b0, n, ex_f, tstep/n_steps)

        #intergration in world frame
        wf, wf_dot = rungekutta_world_frame(wf, bf, tstep/n_steps)

        # if i == 0 and ii == 0:
        #     print 'bf:', bf.T
        #     print 'wf:', wf.T
        
        # print ''
        # print i * 50 + ii
        # print 'bf:', bf.T
        # print 'wf:', wf.T

    t_end = time.time()
    # print 'Time:', t_end-t_start

    # print wf_dot.T


    ##### data save ##########################
    u[i] = e_bf[0] # velocity and angular rate components in the body frame
    v[i] = e_bf[1]
    r[i] = e_bf[2]

    x[i] = wf[0] # position and orientation components in the world frame
    y[i] = wf[1]
    psi[i] = wf[2]

    des_x_v[i] = des_x_vel # shaped refernce velocity
    des_y_v[i] = des_y_vel
    des_psi_v[i] = des_z_ang_vel

    x_v[i] = wf_dot[0] # velocity and angular velocity components in the world frame
    y_v[i] = wf_dot[1]
    psi_v[i] = wf_dot[2]

t_elapsed = time.time() - t_start_
print 't_elapsed:', t_elapsed

# plot of desired and actual translational and rotational velocities
plt.figure(1)
plt.subplot(311)
plt.plot(t_vec,des_x_v,t_vec,x_v) #,title ('x direction'),ylabel('m/s'),legend('desired velocity','real velocity',4)
plt.subplot(312)
plt.plot(t_vec,des_y_v,t_vec,y_v) #title ('y direction'),ylabel('m/s'),legend('desired velocity','real velocity',4)
plt.subplot(313)
plt.plot(t_vec,des_psi_v,t_vec,psi_v) #title ('rotation about z axis'),ylabel('rad/s'),xlabel('time(sec)'),legend('desired velocity','real velocity',4)

plt.show()

# # plot of ArmAssist movement during the control
# plt.figure(2)
# ik = 20 # for plotting every 20*sampling frequency
# for iii in range(len(t_vec)): 
#     if ik == 20:
#         # plot of 3 wheel locations and location of the center mouse camera 
#         plt.plot((x[iii]-L1*np.sin(psi[iii])),(y[iii]+L1*cos(psi[iii])),'mo',(x[iii]+L2*np.sin(32.5/180*pi+psi[iii])),(y[iii]-L2*cos(32.5/180*pi+psi[iii])),'ro',(x[iii]-L2*np.sin(32.5/180*pi-psi[iii])),(y[iii]-L2*cos(32.5/180*pi-psi[iii])),'bo',x[iii],y[iii],'r*')

#         # to draw the trinalge that is made by connecting 3 wheel points
#         line([x[iii]-L1*np.sin(psi[iii]),x[iii]+L2*np.sin(32.5/180*pi+psi[iii]),x[iii]-L2*np.sin(32.5/180*pi-psi[iii]),x[iii]-L1*np.sin(psi[iii])],[y[iii]+L1*cos(psi[iii]),y[iii]-L2*cos(32.5/180*pi+psi[iii]),y[iii]-L2*cos(32.5/180*pi-psi[iii]),y[iii]+L1*cos(psi[iii])])
         
#         ik = 0

#     ik = ik + 1


