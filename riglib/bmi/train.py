'''Needs docs'''

import pickle
import sys

import numpy as np
from scipy.io import loadmat
from plexon import plexfile, psth
from riglib.nidaq import parse

import tables
import kfdecoder
import pdb

def _gen_A(t, s, m, n, off, ndim=3):
    """utility function for generating block-diagonal matrices
    used by the KF
    """
    A = np.zeros([2*ndim+1, 2*ndim+1])
    A_lower_dim = np.array([[t, s], [m, n]])
    A[0:2*ndim, 0:2*ndim] = np.kron(A_lower_dim, np.eye(ndim))
    A[-1,-1] = off
    return np.mat(A)

def _train_KFDecoder_manual_control(cells=None, binlen=0.1, tslice=[None,None], 
    state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    """Train KFDecoder from manual control"""
    # Open plx file
    plx = plexfile.openFile(str(files['plexon']))
    rows = parse.rowbyte(plx.events[:].data)[0][:,0]
    
    lower, upper = 0 < rows, rows < rows.max() + 1
    l, u = tslice
    if l is not None:
        lower = l < rows
    if u is not None:
        upper = rows < u
    tmask = np.logical_and(lower, upper)
    
    #Trim the mask to have exactly an even multiple of 4 worth of data
    if sum(tmask) % 4 != 0:
        midx, = np.nonzero(tmask)
        tmask[midx[-(len(midx) % 4):]] = False
    
    #Grab masked data, remove interpolated data
    h5 = tables.openFile(files['hdf'])
    motion = h5.root.motiontracker
    t, m, d = motion.shape
    motion = motion[np.nonzero(tmask)[0],:,:].reshape(-1, 4, m, d)#motion = motion[np.tile(tmask, [d,m,1]).T].reshape(-1, 4, m, d)
    invalid = np.logical_or(motion[...,-1] == 4, motion[...,-1] < 0)
    motion[invalid] = 0
    kin = motion.sum(axis=1)
    
    # Create PSTH function
    if cells == None: cells = plx.units
    units = np.array(cells).astype(np.int32)
    spike_bin_fn = psth.SpikeBin(units, binlen)
    
    neurows = rows[tmask][3::4]
    neurons = np.array(list(plx.spikes.bin(neurows, spike_bin_fn)))
    mFR = np.mean(neurons,axis=0)
    sdFR = np.std(neurons,axis=0)
    if len(kin) != len(neurons):
        raise ValueError('Training data and neural data are the wrong length: %d vs. %d'%(len(kin), len(neurons)))
    
    # Match kinematics with the task state
    task_states = ['origin_hold', 'terminus', 'terminus_hold', 'reward']
    states = h5.root.motiontracker_msgs[:]
    # assign names to each of the states
    state_numbering = dict((n, i) for i, n in enumerate(np.unique(states['msg'])))
    
    # event sequence of interest
    trial_seq = np.array([state_numbering[n] for n in task_states])
    idx = np.convolve(trial_seq, trial_seq, 'valid')
    
    event_seq = np.array([state_numbering[n] for n in states['msg']])
    found = np.convolve(event_seq, trial_seq, 'valid') == idx
    
    times = states[found]['time']
    if len(times) % 2 == 1:
        times = times[:-1]
    slices = times.reshape(-1, 2)
    t, m, d = h5.root.motiontracker.shape
    mask = np.ones((t/4, m, d), dtype=bool)
    for s, e in slices:
        mask[s/4:e/4] = False
    kin = np.ma.array(kin, mask=mask[tmask[3::4]])
    
    # calculate velocity
    kin[(kin[...,:3] == 0).all(-1)] = np.ma.masked
    kin[kin[...,-1] < 0] = np.ma.masked
    velocity = np.ma.diff(kin[...,:3], axis=0)*60
    kin = np.ma.hstack([kin[:-1,:,:3], velocity])
    
    hand_kin = kin[:, [14,kin.shape[1]/2+14], :]
    hand_kin = hand_kin.reshape(len(hand_kin), -1)
    
    # train KF model parameters
    neurons = neurons.T
    n_neurons = neurons.shape[0]
    hand_kin = hand_kin.T

    hand_kin_vars = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']

    train_vars = stochastic_vars[:]
    try: train_vars.remove('offset')
    except: pass 

    try:
        state_inds = [hand_kin_vars.index(x) for x in state_vars]
        stochastic_inds = [hand_kin_vars.index(x) for x in stochastic_vars]
    	train_inds = [hand_kin_vars.index(x) for x in train_vars]
        stochastic_state_inds = [state_vars.index(x) for x in stochastic_vars]
    except:
        raise ValueError("Invalid kinematic variable(s) specified for KFDecoder state")
    C = np.zeros([n_neurons, len(state_inds)])
    C[:, stochastic_state_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(hand_kin[train_inds, :], neurons[:,:-1])
    
    Delta_KINARM = 1./10
    Delta_BMI3D = 1./60
    loop_update_ratio = Delta_BMI3D/Delta_KINARM
    A = _gen_A(1, 1./60, 0, 0.8**loop_update_ratio, 1, ndim=3)[np.ix_(state_inds, state_inds)]
    W = _gen_A(0, 0, 0, 700*loop_update_ratio, 0, ndim=3)[np.ix_(state_inds, state_inds)]
    
    # instantiate low-level kf
    unit_inds, = np.nonzero(np.array(C)[:,-1])
    ## TODO remove next line and make user option
    is_stochastic = np.array([False, False, True, True, False])
    kf = kfdecoder.KalmanFilter(A, W, C[unit_inds,:], Q[np.ix_(unit_inds,unit_inds)], is_stochastic=is_stochastic)
    units = units[unit_inds,:]
    mFR = mFR[unit_inds]
    sdFR = sdFR[unit_inds]

    # instantiate KFdecoder
    bounding_box = np.array([-250., -140.]), np.array([250., 140.])
    states_to_bound = ['hand_px', 'hand_pz']
    decoder = kfdecoder.KFDecoder(kf, mFR, sdFR, units, bounding_box, state_vars, states_to_bound)
    return decoder


def _train_KFDecoder_visual_feedback(cells=None, binlen=0.1, tslice=[None,None], 
    state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    """Train KFDecoder from visual feedback of cursor movement"""
    # Open plx file
    plx = plexfile.openFile(str(files['plexon']))
    rows = parse.rowbyte(plx.events[:].data)[1][:,0]
    
    lower, upper = 0 < rows, rows < rows.max() + 1
    l, u = tslice
    if l is not None:
        lower = l < rows
    if u is not None:
        upper = rows < u
    tmask = np.logical_and(lower, upper)
    
    #Grab masked kinematic data
    h5 = tables.openFile(files['hdf'])
    cursor = h5.root.task[:]['cursor']
    kin = cursor[tmask,:]
    
    # Create PSTH function
    if cells == None: cells = plx.units
    units = np.array(cells).astype(np.int32)
    spike_bin_fn = psth.SpikeBin(units, binlen)
    
    neurows = rows[tmask]
    neurons = np.array(list(plx.spikes.bin(neurows, spike_bin_fn)))
    mFR = np.mean(neurons,axis=0)
    sdFR = np.std(neurons,axis=0)
    if len(kin) != len(neurons):
        raise ValueError('Training data and neural data are the wrong length: %d vs. %d'%(len(kin), len(neurons)))
    
    # calculate cursor velocity
    velocity = np.diff(kin, axis=0)*60
    kin = np.hstack([kin, velocity])
    
    # train KF model parameters
    neurons = neurons.T
    n_neurons = neurons.shape[0]
    kin = kin.T

    kin_vars = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']

    train_vars = stochastic_vars[:]
    try: train_vars.remove('offset')
    except: pass 

    try:
        state_inds = [kin_vars.index(x) for x in state_vars]
        stochastic_inds = [kin_vars.index(x) for x in stochastic_vars]
        train_inds = [kin_vars.index(x) for x in train_vars]
        stochastic_state_inds = [state_vars.index(x) for x in stochastic_vars]
    except:
        raise ValueError("Invalid kinematic variable(s) specified for KFDecoder state")
    C = np.zeros([n_neurons, len(state_inds)])
    C[:, stochastic_state_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(kin[train_inds, :], neurons[:,:-1])
    
    Delta_KINARM = 1./10
    Delta_BMI3D = 1./60
    loop_update_ratio = Delta_BMI3D/Delta_KINARM
    A = _gen_A(1, 1./60, 0, 0.8**loop_update_ratio, 1, ndim=3)[np.ix_(state_inds, state_inds)]
    W = _gen_A(0, 0, 0, 700*loop_update_ratio, 0, ndim=3)[np.ix_(state_inds, state_inds)]
    
    # instantiate low-level kf
    unit_inds, = np.nonzero(np.array(C)[:,-1])
    ## TODO remove next line and make user option
    is_stochastic = np.array([False, False, True, True, False])
    kf = kfdecoder.KalmanFilter(A, W, C[unit_inds,:], Q[np.ix_(unit_inds,unit_inds)], is_stochastic=is_stochastic)
    units = units[unit_inds,:]
    mFR = mFR[unit_inds]
    sdFR = sdFR[unit_inds]

    # instantiate KFdecoder
    bounding_box = np.array([-250., -140.]), np.array([250., 140.])
    states_to_bound = ['hand_px', 'hand_pz']
    decoder = kfdecoder.KFDecoder(kf, mFR, sdFR, units, bounding_box, state_vars, states_to_bound)
    return decoder

# def _train_KFDecoder_brain_control(cells=None, binlen=0.1, tslice=[None,None], 
#     state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], 
#     stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
#     """Train KFDecoder from brain control"""

#     #Grab masked data, remove interpolated data
#     h5 = tables.openFile(files['hdf'])
#     spike_counts = h5.root.task[:]['bins']
#     kin = h5.root.task[:]['cursor']
#     print '!!!!!!!!!!!!!!'

#     hand_kin_vars = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
#     return h5

#     train_vars = stochastic_vars[:]
#     if 'offset' in train_vars: train_vars.remove('offset')

#     try:
#         state_inds = [hand_kin_vars.index(x) for x in state_vars]
#         stochastic_inds = [hand_kin_vars.index(x) for x in stochastic_vars]
#     	train_inds = [hand_kin_vars.index(x) for x in train_vars]
#         stochastic_state_inds = [state_vars.index(x) for x in stochastic_vars]
#     except:
#         raise ValueError("Invalid kinematic variable(s) specified for KFDecoder state")
#     C = np.zeros([n_neurons, len(state_inds)])
#     C[:, stochastic_state_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(hand_kin[train_inds, :], neurons[:,:-1])

#     Delta_KINARM = 1./10
#     Delta_BMI3D = 1./60
#     loop_update_ratio = Delta_BMI3D/Delta_KINARM
#     A = _gen_A(1, 1./60, 0, 0.8**loop_update_ratio, 1, ndim=3)[np.ix_(state_inds, state_inds)]
#     W = _gen_A(0, 0, 0, 700*loop_update_ratio, 0, ndim=3)[np.ix_(state_inds, state_inds)]
    
#     # instantiate low-level kf
#     unit_inds, = np.nonzero(np.array(C)[:,-1])
#     kf = kfdecoder.KalmanFilter(A, W, C[unit_inds,:], Q[np.ix_(unit_inds,unit_inds)])
#     units = units[unit_inds,:]

#     # instantiate KFdecoder
#     bounding_box = np.array([-100., -100.]), np.array([100., 100.])
#     states_to_bound = ['hand_px', 'hand_pz']
#     decoder = kfdecoder.KFDecoder(kf, None, None, units, bounding_box, state_vars, states_to_bound)
#     return decoder

def _train_KFDecoder_2D_sim(is_stochastic, drives_neurons, units, 
    bounding_box, states_to_bound, include_y=True, dt=0.1):
    # TODO options to resample the state-space model at different update rates
    v = 0.8
    n_neurons = units.shape[0]
    if include_y:
        state_vars = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
        A = np.array([[1, 0, 0, dt, 0, 0,  0],
                      [0, 1, 0, 0,  0, 0,  0],
                      [0, 0, 1, 0,  0, dt, 0],
                      [0, 0, 0, v,  0, 0,  0],
                      [0, 0, 0, 0,  0, 0,  0],
                      [0, 0, 0, 0,  0, v,  0],
                      [0, 0, 0, 0,  0, 0,  1]])
    else:
        state_vars = ['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset']
        A = np.array([[1, 0, dt, 0, 0],
                      [0, 1, 0, dt, 0],
                      [0, 0, v,  0, 0],
                      [0, 0, 0,  v, 0],
                      [0, 0, 0,  0, 1]])

    nX = A.shape[0]
    w = 1e-3
    W = np.diag(w * np.ones(nX))
    W[np.ix_(~is_stochastic, ~is_stochastic)] = 0

    C = np.random.standard_normal([n_neurons, nX])
    C[:, ~drives_neurons] = 0

    Q = 10 * np.identity(n_neurons) 

    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=is_stochastic)
    kf.alt = False

    decoder = kfdecoder.KFDecoder(kf, None, None, units, bounding_box, state_vars, states_to_bound)
    return decoder

if __name__ == '__main__':
    test_mc = False
    test_bc = True 
    if test_mc:
        block = 'cart20130428_01'
        #block = 'cart20130425_05'
        files = dict(plexon='/storage/plexon/%s.plx' % block, hdf='/storage/rawdata/hdf/%s.hdf' % block)
        binlen = 0.1
        tslice = [1., 300.]
        
        decoder = _train_KFDecoder_manual_control(cells=None, binlen=0.1, tslice=[None,None],
            state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], **files) 
    if test_bc:
        block = 'cart20130521_04'
        files = dict(hdf='/storage/rawdata/hdf/%s.hdf' % block)
        binlen = 0.1
        tslice = [1., 300.]
        
        decoder = _train_KFDecoder_brain_control(cells=None, binlen=0.1, tslice=[None,None],
            state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], **files) 
