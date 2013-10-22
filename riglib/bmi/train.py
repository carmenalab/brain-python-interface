'''
Methods to create and train Decoder objects
'''

import pickle
import sys

import numpy as np
from scipy.io import loadmat
from plexon import plexfile, psth
from riglib.nidaq import parse

import tables
import kfdecoder, ppfdecoder
import pdb
from . import state_space_models

def _train_KFDecoder_manual_control(cells=None, binlen=0.1, tslice=[None,None], 
    state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    """Train KFDecoder from manual control"""
    # Open plx file
    plx = plexfile.openFile(str(files['plexon']))
    events = plx.events[:].data
    syskey=0
    # get system registrations
    reg = parse.registrations(events)
    # find the key for the motiontracker system data
    for key, system in reg.items():
        if (system == 'otiontracker') or (system == 'motiontracker'): #yes this typo is intentional! we could never figure out why the first character doesn't get recorded in the registration
            syskey = key
    # get the corresponding hdf rows
    rows = parse.rowbyte(events)[syskey][:,0]
    
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
    task_states = ['target', 'hold']
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
    hand_kin = hand_kin.reshape(len(hand_kin), -1)/10.0 #convert motiontracker kinematic data to cm
    
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
    
    # State-space model set from expert data
    A, W = state_space_models.linear_kinarm_kf(update_rate=1./60)
    A = A[np.ix_(state_inds, state_inds)]
    W = W[np.ix_(state_inds, state_inds)]
    
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
    decoder = kfdecoder.KFDecoder(kf, mFR, sdFR, units, bounding_box, state_vars, states_to_bound, binlen=binlen)
    return decoder


def _train_KFDecoder_visual_feedback(cells=None, binlen=0.1, tslice=[None,None], 
    state_vars=['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    update_rate=binlen
    # Open plx file
    plx = plexfile.openFile(str(files['plexon']))
    # pull out event data
    events = plx.events[:].data
    # get system registrations
    reg = parse.registrations(events)
    # find the key for the task system data
    for key, system in reg.items():
        if (system[0] == 'ask') or (system[0] == 'task'):
            syskey = key
    # get the corresponding hdf rows
    rows = parse.rowbyte(events)[syskey][:,0]
    
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

    #Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
    # and select correct bins for neural data
    step = int(binlen/(1./60))
    kin = kin[::step, :]
    neurows = neurows[::step]

    neurons = np.array(list(plx.spikes.bin(neurows, spike_bin_fn)))
    print len(neurons)
    mFR = np.mean(neurons,axis=0)
    sdFR = np.std(neurons,axis=0)
    if len(kin) != len(neurons):
        raise ValueError('Training data and neural data are the wrong length: %d vs. %d'%(len(kin), len(neurons)))
    
    # calculate cursor velocity
    velocity = np.diff(kin, axis=0)*(1/binlen)
    kin = np.hstack([kin[1:], velocity])
    
    # train KF model parameters
    neurons = neurons
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
    
    # State-space model set from expert data
    A, W = state_space_models.linear_kinarm_kf(update_rate=update_rate)
    A = A[np.ix_(state_inds, state_inds)]
    W = W[np.ix_(state_inds, state_inds)]
    
    # instantiate low-level kf
    unit_inds, = np.nonzero(np.array(C)[:,-1])
    C = C[unit_inds,:]
    Q = Q[np.ix_(unit_inds,unit_inds)]
    print len(unit_inds)
    is_stochastic = np.array([x in stochastic_vars for x in state_vars])
    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=is_stochastic)
    units = units[unit_inds,:]
    mFR = mFR[unit_inds]
    sdFR = sdFR[unit_inds]

    # instantiate KFdecoder
    #bounding_box = np.array([-250., -140.]), np.array([250., 140.])
    bounding_box = np.array([-25., -14.]), np.array([25., 14.]) # bounding box in cm
    states_to_bound = ['hand_px', 'hand_pz']
    neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
    drives_neurons = np.array([x in neuron_driving_states for x in state_vars])
    decoder = kfdecoder.KFDecoder(kf, mFR, sdFR, units, bounding_box, 
        state_vars, drives_neurons, states_to_bound, binlen=binlen, 
        tslice=tslice)

    from clda import KFRML
    n_neurons, n_states = C.shape
    print n_neurons
    print neurons.shape
    R = np.mat(np.zeros([n_states, n_states]))
    S = np.mat(np.zeros([n_neurons, n_states]))
    R_small, S_small, T = KFRML.compute_suff_stats(kin[train_inds, :], neurons[unit_inds,:-1])
    ## print R.shape
    ## print stochastic_state_inds
    ## print R_small.shape
    ## print S.shape
    ## print stochastic_state_inds
    ## print S_small.shape

    R[np.ix_(stochastic_state_inds, stochastic_state_inds)] = R_small
    print S.shape
    print S_small.shape
    S[:,stochastic_state_inds] = S_small
    
    decoder.kf.R = R
    decoder.kf.S = S
    decoder.kf.T = T 

    
    return decoder

def _train_KFDecoder_visual_feedback_old(cells=None, binlen=0.1, tslice=[None,None], 
    state_vars=['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    update_rate=binlen
    # Open plx file
    plx = plexfile.openFile(str(files['plexon']))
    # pull out event data
    events = plx.events[:].data
    # get system registrations
    reg = parse.registrations(events)
    # find the key for the task system data
    for key, system in reg.items():
        if (system[0] == 'ask') or (system[0] == 'task'):
            syskey = key
    # get the corresponding hdf rows
    rows = parse.rowbyte(events)[syskey][:,0]
    
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
    kin = np.hstack([kin[1:], velocity])
    
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
    
    # State-space model set from expert data
    A, W = state_space_models.linear_kinarm_kf(update_rate=update_rate)
    A = A[np.ix_(state_inds, state_inds)]
    W = W[np.ix_(state_inds, state_inds)]
    
    # instantiate low-level kf
    unit_inds, = np.nonzero(np.array(C)[:,-1])
    is_stochastic = np.array([x in stochastic_vars for x in state_vars])
    kf = kfdecoder.KalmanFilter(A, W, C[unit_inds,:], Q[np.ix_(unit_inds,unit_inds)], is_stochastic=is_stochastic)
    units = units[unit_inds,:]
    mFR = mFR[unit_inds]
    sdFR = sdFR[unit_inds]

    # instantiate KFdecoder
    bounding_box = np.array([-250., -140.]), np.array([250., 140.])
    states_to_bound = ['hand_px', 'hand_pz']
    neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
    drives_neurons = np.array([x in neuron_driving_states for x in state_vars])
    decoder = kfdecoder.KFDecoder(kf, mFR, sdFR, units, bounding_box, state_vars, drives_neurons, states_to_bound, binlen=binlen)

    from clda import KFRML
    R, S, T = KFRML.compute_suff_stats(kin[train_inds, :], neurons[:,:-1])
    decoder.kf.R = R
    decoder.kf.S = S
    decoder.kf.T = T 

    return decoder

def _train_PPFDecoder_2D_sim(stochastic_states, neuron_driving_states, units,
    bounding_box, states_to_bound, include_y=True, dt=0.1, v=0.4):
    '''
    Train a simulation PPFDecoder
    '''
    raise NotImplementedError

def _train_KFDecoder_2D_sim(stochastic_states, neuron_driving_states, units,
    bounding_box, states_to_bound, include_y=True, dt=0.1, v=0.8):
    # TODO options to resample the state-space model at different update rates
    n_neurons = units.shape[0]
    if include_y:
        states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
        A = np.array([[1, 0, 0, dt, 0,  0,  0],
                      [0, 1, 0, 0,  dt, 0,  0],
                      [0, 0, 1, 0,  0,  dt, 0],
                      [0, 0, 0, v,  0,  0,  0],
                      [0, 0, 0, 0,  v,  0,  0],
                      [0, 0, 0, 0,  0,  v,  0],
                      [0, 0, 0, 0,  0,  0,  1]])
    else:
        states = ['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset']
        A = np.array([[1, 0, dt, 0, 0],
                      [0, 1, 0, dt, 0],
                      [0, 0, v,  0, 0],
                      [0, 0, 0,  v, 0],
                      [0, 0, 0,  0, 1]])

    drives_neurons = np.array([x in neuron_driving_states for x in states])
    is_stochastic = np.array([x in stochastic_states for x in states])

    nX = A.shape[0]
    w = 0.0007
    W = np.diag(w * np.ones(nX))
    W[np.ix_(~is_stochastic, ~is_stochastic)] = 0

    C = np.random.standard_normal([n_neurons, nX])
    C[:, ~drives_neurons] = 0

    #C *= 6

    Q = 10 * np.identity(n_neurons) 
    # set det(Q) to be ~10^10
    #Q = 100 * np.identity(n_neurons) 

    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=is_stochastic)
    kf.alt = False

    mFR = 0
    sdFR = 1

    decoder = kfdecoder.KFDecoder(kf, mFR, sdFR, units, bounding_box, 
        states, drives_neurons, states_to_bound)

    cm_to_m = 0.01
    decoder.kf.R = np.mat(np.identity(decoder.kf.C.shape[1]))
    decoder.kf.S = decoder.kf.C * cm_to_m
    decoder.kf.T = decoder.kf.Q + decoder.kf.S*decoder.kf.S.T

    return decoder

def load_from_mat_file(decoder_fname, bounding_box=None, 
    states=['p_x', 'p_y', 'v_x', 'v_y', 'off'], states_to_bound=[]):
    """Create KFDecoder from MATLAB decoder file used in a Dexterit-based
    BMI
    """
    decoder_data = loadmat(decoder_fname)['decoder']
    A = decoder_data['A'][0,0]
    W = decoder_data['W'][0,0]
    H = decoder_data['H'][0,0]
    Q = decoder_data['Q'][0,0]
    mFR = decoder_data['mFR'][0,0]
    sdFR = decoder_data['sdFR'][0,0]

    pred_sigs = [str(x[0]) for x in decoder_data['predSig'][0,0].ravel()]
    unit_lut = {'a':1, 'b':2, 'c':3, 'd':4}
    units = [(int(sig[3:6]), unit_lut[sig[-1]]) for sig in pred_sigs]

    kf = KalmanFilter(A, W, H, Q)
    kfdecoder = KFDecoder(kf, mFR, sdFR, units, bounding_box, states, states_to_bound)

    return kfdecoder

def rescale_KFDecoder_units(dec, scale_factor=10):
    '''
    Convert the units of a KFDecoder, e.g. from mm to cm

    C and W matrices of KalmanFilter must be updated for the new units. 
    A and Q are unitless and thus remain the same

    Parameters
    ----------
    dec : KFDecoder instance
        KFDecoder object
    scale_factor : numerical
        defines how much bigger the new unit is than the old one
    '''
    inds = np.nonzero((np.diag(dec.kf.W) > 0) * dec.drives_neurons)[0]
    nS = dec.kf.W.shape[0]
    S_diag = np.ones(nS)
    S_diag[inds] = scale_factor
    S = np.mat(np.diag(S_diag))
    #S = np.mat(np.diag([1., 1, 1, 10, 10, 10, 1]))
    dec.kf.C *= S
    dec.kf.W *= S.I * S.I
    try:
        dec.kf.C_xpose_Q_inv_C = S.T * dec.kf.C_xpose_Q_inv_C * S
        dec.kf.C_xpose_Q_inv = S.T * dec.kf.C_xpose_Q_inv
    except:
        pass
    dec.bounding_box = tuple([x / scale_factor for x in dec.bounding_box])
    return dec

def convert_KFDecoder_to_PPFDecoder(dec):
    binlen = dec.binlen
    beta = dec.kf.C / binlen

    dt = 1./180
    A, W = state_space_models.linear_kinarm_kf(update_rate=dt, units_mult=0.01)
    args = (dec.bounding_box, dec.states, dec.drives_neurons, dec.states_to_bound)
    ppf = ppfdecoder.PointProcessFilter(A, W, beta, dt)
    dec_ppf = ppfdecoder.PPFDecoder(ppf, dec.units, *args)
    dec_ppf.n_subbins = 3
    return dec_ppf

def inflate(A, current_states, full_state_ls, axis=0):
    '''
    'Inflate' a matrix by filling in rows/columns with zeros
    '''
    nS = len(full_state_ls)#A.shape[0]
    if axis == 0:
        A_new = np.zeros([nS, A.shape[1]])
    elif axis == 1:
        A_new = np.zeros([A.shape[0], nS])

    new_inds = [full_state_ls.index(x) for x in current_states]
    if axis == 0:
        A_new[new_inds, :] = A
    elif axis == 1:
        A_new[:, new_inds] = A

    return A_new

def _train_PPFDecoder_sim_known_beta(beta, units, dt=0.005, dist_units='m'):
    '''
    Create a PPFDecoder object to decode 2D velocity from a known 'beta' matrix
    '''
    units_mult_lut = dict(m=1., cm=0.01)
    units_mult = units_mult_lut[dist_units]

    A, W = state_space_models.linear_kinarm_kf(update_rate=dt, units_mult=units_mult)

    bounding_box = (np.array([-0.25, -0.14])/units_mult, np.array([0.25, 0.14])/units_mult)
    drives_neurons = ['hand_vx', 'hand_vz', 'offset']
    states_to_bound = ['hand_px', 'hand_pz']
    states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
    args = (bounding_box, states, drives_neurons, states_to_bound)
    
    ppf = ppfdecoder.PointProcessFilter(A, W, beta, dt)
    dec = ppfdecoder.PPFDecoder(ppf, units, *args)

    # Force decoder to run at max 60 Hz
    dec.bmicount = 0
    dec.bminum = 0
    return dec

