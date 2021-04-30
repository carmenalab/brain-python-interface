#!/usr/bin/python 
'''
Script to map between new and old assists
infinite horizon LQR feedback controllers are hackishly fit based on arrival times
'''
import numpy as np
from riglib.bmi import feedback_controllers, kfdecoder, ppfdecoder
import matplotlib.pyplot as plt
from db import dbfunctions as dbfn

dec = dbfn.get_decoder(2326)

if isinstance(dec, kfdecoder.KFDecoder):
    dt = 1./10
    dec_type = 'kf'
elif isinstance(dec, ppfdecoder.PPFDecoder):
    dt = 1./180
    dec_type = 'ppf'

eff_targ_radius = 1.2

I = np.eye(3)
B = np.bmat([[0*I], 
              [dt/1e-3 * I],
              [np.zeros([1, 3])]])
A = dec.filt.A

## F = []
## F.append(np.zeros([3, 7]))
## for k in range(num_assist_levels):
## 
##     F_k = np.array(feedback_controllers.LQRController.dlqr(A, B, Q, R, eps=1e-15))
##     F.append(F_k)
## 
## F_assist = np.dstack([np.array(x) for x in F]).transpose([2,0,1])

def calc_F(scale_factor):
    tau_scale = (28.*28)/1000/3 * scale_factor # np.array([3, 2.5, 1.5])
    
    w_x = 1;
    w_v = 3*tau_scale**2/2;
    w_r = 1e6*tau_scale**4;
    
    Q = np.mat(np.diag([w_x, w_x, w_x, w_v, w_v, w_v, 0]))
    R = np.mat(np.diag([w_r, w_r, w_r]))
    F = feedback_controllers.LQRController.dlqr(A, B, Q, R, eps=1e-15)
    return F

def calc_traj(A, B, F, n_steps=10000):
    x = np.mat(np.zeros([A.shape[0], n_steps]))
    x[:,0] = np.mat([10., 0, 0, 0, 0, 0, 1]).reshape(-1,1) # start at (10,0,0) with zero vel

    target_state = np.mat([0., 0, 0, 0, 0, 0, 1]).reshape(-1,1)

    for k in range(1, n_steps):
        x[:,k] = (A-B*F)*x[:,k-1] + B*F*(target_state - x[:,k-1])
    return x

def _calc(A, B, F, n_steps=10000):
    x = np.mat([10., 0, 0, 0, 0, 0, 1]).reshape(-1,1) # start at (10,0,0) with zero vel

    target_state = np.mat([0., 0, 0, 0, 0, 0, 1]).reshape(-1,1)

    idx = 0
    while x[0,0] > 1.2:
        x = (A-B*F)*x + B*F*(target_state - x)
        idx += 1
    return idx

def calc_arrival_time(scale_factor):
    F = calc_F(scale_factor)
    arrival_idx = np.array(_calc(A, B, F))
    return arrival_idx

scale_factors = np.linspace(1., 15, 500)
arrival_time = np.zeros(len(scale_factors))
for k, scale in enumerate(scale_factors):
    print(k)
    arrival_time[k] = calc_arrival_time(scale) * dt


scale_factor = 2
mean_speeds = scale_factor * 5 * np.linspace(0, 1, 20)
old_arrival_times = 8.8 / mean_speeds

plt.close('all')
plt.figure()
ax = plt.subplot(111)
ax.hold(True)
ax.plot(scale_factors, arrival_time)

for t in old_arrival_times:
    ax.plot([1, 9], [t, t])
    #ax.axhline(t, 1, 9)

plt.show()

fn = lambda t: np.nonzero((arrival_time[1:] > t) & (arrival_time[:-1] < t))[0][0]
inds = np.array(list(map(fn, old_arrival_times[1:])))
scale_factors_assist = 0.5 * (scale_factors[inds] + scale_factors[inds+1])

F = []
F.append(np.zeros([3, 7]))
F += list(map(calc_F, scale_factors_assist))
F_assist = np.dstack([np.array(x) for x in F]).transpose([2,0,1])

import pickle
pickle.dump(F_assist, open('/storage/assist_params/assist_%dlevels_%s.pkl' % (F_assist.shape[0], dec_type), 'w'))

## n_steps = 2000
## traj = np.array(calc_traj(A, B, F, n_steps=n_steps))
## print traj
## 
## plt.figure()
## ax = plt.subplot(111)
## ax.hold(True)
## ax.plot(traj[0,:])
## ax.axhline(1.2, 0, n_steps, color='green')
## plt.show()
