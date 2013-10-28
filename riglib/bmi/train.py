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
from itertools import izip

def _train_KFDecoder_manual_control(cells=None, binlen=0.1, tslice=[None,None], 
    state_vars=['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    #state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], 
    #stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    """
    Train KFDecoder from manual control
    """
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

def _get_tmask(plx_fname, tslice, syskey_fn=lambda x: x[0] in ['task', 'ask']):
    # Open plx file
    plx = plexfile.openFile(plx_fname)
    events = plx.events[:].data
    syskey=0
    # get system registrations
    reg = parse.registrations(events)
    # find the key for the motiontracker system data
    for key, system in reg.items():
        if syskey_fn(system):
            syskey = key
            break
    # get the corresponding hdf rows
    rows = parse.rowbyte(events)[syskey][:,0]
    
    lower, upper = 0 < rows, rows < rows.max() + 1
    l, u = tslice
    if l is not None:
        lower = l < rows
    if u is not None:
        upper = rows < u
    tmask = np.logical_and(lower, upper)
    return tmask, rows

def _train_PPFDecoder_visual_feedback(cells=None, binlen=1./180, tslice=[None,None], 
    state_vars=['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    '''
    Train a PPFDecoder from visual feedback
    '''
    binlen = 1./180 # TODO remove hardcoding!
    update_rate=binlen
    # Open plx file
    plx_fname = str(files['plexon'])
    plx = plexfile.openFile(plx_fname)
    tmask, rows = _get_tmask(plx_fname, tslice, syskey_fn=lambda x: x[0] in ['task', 'ask'])
    
    #Grab masked kinematic data
    h5 = tables.openFile(files['hdf'])
    kin = h5.root.task[:]['cursor']
    inds, = np.nonzero(tmask)
    step_fl = binlen/(1./60) 
    if step_fl < 1: # more than one spike bin per kinematic obs
        n_repeats = int((1./60)/binlen)
        inds = np.sort(np.hstack([inds]*n_repeats))
    else:
        step = int(binlen/(1./60)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
        inds = inds[::step] #slice(None, None, step)
    kin = kin[inds]

    ## Bin the neural data
    cells = np.unique(cells)
    if cells == None: cells = plx.units # Use all of the units if none are specified
    units = np.array(cells).astype(np.int32)
    spike_bin_fn = psth.SpikeBin(units, binlen)
    neurows = rows[tmask]
    
    # interpolate between the rows to 180 Hz
    interp_rows = []
    neurows = np.hstack([neurows[0] - 1./60, neurows])
    for r1, r2 in izip(neurows[:-1], neurows[1:]):
        interp_rows += list(np.linspace(r1, r2, 4)[1:])
    interp_rows = np.array(interp_rows)

    spike_counts = np.array(list(plx.spikes.bin(interp_rows, spike_bin_fn)))

    if len(kin) != len(spike_counts):
        raise ValueError('Training data and neural data are the wrong length: %d vs. %d'%(len(kin), len(spike_counts)))
    
    # calculate cursor velocity
    velocity = np.diff(kin, axis=0) * 1./binlen
    kin = np.hstack([kin[1:], velocity])
    spike_counts = spike_counts.T
    spike_counts = spike_counts[:,:-1]

    return train_endpt_velocity_PPFDecoder(kin, spike_counts, units, state_vars, stochastic_vars, update_rate=binlen, tslice=tslice)

def train_endpt_velocity_PPFDecoder(kin, spike_counts, units, state_vars, stochastic_vars, update_rate=0.1, tslice=None):
    '''
    Train a Point-process filter decoder which predicts the endpoint velocity
    '''
    binlen = update_rate
    n_neurons = spike_counts.shape[0]
    kin = kin.T

    kin_vars = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']

    # C and Q should be trained on all of the stochastic state variables, excluding the offset terms
    train_vars = stochastic_vars[:]
    if 'offset' in train_vars: train_vars.remove('offset')

    try:
        state_inds = [kin_vars.index(x) for x in state_vars]
        stochastic_inds = [kin_vars.index(x) for x in stochastic_vars]
        train_inds = [kin_vars.index(x) for x in train_vars]
        stochastic_state_inds = [state_vars.index(x) for x in stochastic_vars]
    except:
        raise ValueError("Invalid kinematic variable(s) specified for KFDecoder state")

    C = np.zeros([n_neurons, len(state_inds)])
    C[:, stochastic_state_inds], pvals = ppfdecoder.PointProcessFilter.MLE_obs_model(kin[train_inds, :], spike_counts)
    
    # TODO Eliminate units which have baseline rates of 0 (w.p. 1, no spikes are observed for these units)

    # State-space model set from expert data
    A, W = state_space_models.linear_kinarm_kf(update_rate=update_rate)
    if len(state_inds) < len(kin_vars): # Only a subset of states are represented in the decoder
        A = A[np.ix_(state_inds, state_inds)]
        W = W[np.ix_(state_inds, state_inds)]
    
    # Control input matrix for SSM for control inputs
    I = np.mat(np.eye(3))
    B = np.vstack([I, update_rate*1000 * I, np.zeros([1,3])])

    # instantiate Decoder
    is_stochastic = np.array([x in stochastic_vars for x in state_vars])
    ppf = ppfdecoder.PointProcessFilter(
            A, W, C, dt=update_rate, is_stochastic=is_stochastic, B=B)


    bounding_box = np.array([-25., -14.]), np.array([25., 14.]) # bounding box in cm
    states_to_bound = ['hand_px', 'hand_pz']
    neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
    drives_neurons = np.array([x in neuron_driving_states for x in state_vars])
    decoder = ppfdecoder.PPFDecoder(ppf, units, bounding_box, 
        state_vars, drives_neurons, states_to_bound, binlen=binlen, 
        tslice=tslice)

    return decoder

def _train_KFDecoder_visual_feedback(cells=None, binlen=0.1, tslice=[None,None], 
    state_vars=['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    '''
    Train a KFDecoder from visual feedback
    '''
    update_rate=binlen
    # Open plx file
    plx_fname = str(files['plexon'])
    plx = plexfile.openFile(plx_fname)
    tmask, rows = _get_tmask(plx_fname, tslice, syskey_fn=lambda x: x[0] in ['task', 'ask'])
    tmask_continuous = np.array_equal(np.unique(np.diff(np.nonzero(tmask)[0])), np.array([1]))
    
    #Grab masked kinematic data
    h5 = tables.openFile(files['hdf'])
    kin = h5.root.task[tmask]['cursor']
    step = int(binlen/(1./60)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
    kin = kin[::step, :]

    ## Bin the neural data
    if cells == None: cells = plx.units # Use all of the units if none are specified
    units = np.array(cells).astype(np.int32)
    spike_bin_fn = psth.SpikeBin(units, binlen)
    neurows = rows[tmask]
    neurows = neurows[::step]
    spike_counts = np.array(list(plx.spikes.bin(neurows, spike_bin_fn)))

    if len(kin) != len(spike_counts):
        raise ValueError('Training data and neural data are the wrong length: %d vs. %d'%(len(kin), len(spike_counts)))
    
    # calculate cursor velocity
    velocity = np.diff(kin, axis=0) * 1./binlen
    kin = np.hstack([kin[1:], velocity])
    spike_counts = spike_counts.T
    spike_counts = spike_counts[:,:-1]
    return train_endpt_velocity_KFDecoder(kin, spike_counts, units, state_vars, stochastic_vars, update_rate=binlen, tslice=tslice)

def train_endpt_velocity_KFDecoder(kin, spike_counts, units, state_vars, stochastic_vars, update_rate=0.1, tslice=None):
    binlen = update_rate
    n_neurons = spike_counts.shape[0]
    kin = kin.T

    kin_vars = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']

    # C and Q should be trained on all of the stochastic state variables, excluding the offset terms
    train_vars = stochastic_vars[:]
    if 'offset' in train_vars: train_vars.remove('offset')

    try:
        state_inds = [kin_vars.index(x) for x in state_vars]
        stochastic_inds = [kin_vars.index(x) for x in stochastic_vars]
        train_inds = [kin_vars.index(x) for x in train_vars]
        stochastic_state_inds = [state_vars.index(x) for x in stochastic_vars]
    except:
        raise ValueError("Invalid kinematic variable(s) specified for KFDecoder state")

    C = np.zeros([n_neurons, len(state_inds)])
    C[:, stochastic_state_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(kin[train_inds, :], spike_counts)
    
    # Eliminate units which have baseline rates of 0 (w.p. 1, no spikes are observed for these units)
    unit_inds, = np.nonzero(np.array(C)[:,-1])
    C = C[unit_inds,:]
    Q = Q[np.ix_(unit_inds, unit_inds)]
    units = units[unit_inds,:]

    mFR = np.mean(spike_counts[unit_inds, :], axis=1)
    sdFR = np.std(spike_counts[unit_inds, :], axis=1)
    #mFR = mFR[unit_inds]
    #sdFR = sdFR[unit_inds]

    # State-space model set from expert data
    A, W = state_space_models.linear_kinarm_kf(update_rate=update_rate)
    if len(state_inds) < len(kin_vars): # Only a subset of states are represented in the decoder
        A = A[np.ix_(state_inds, state_inds)]
        W = W[np.ix_(state_inds, state_inds)]
    
    # instantiate KFdecoder
    is_stochastic = np.array([x in stochastic_vars for x in state_vars])
    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=is_stochastic)
    bounding_box = np.array([-25., -14.]), np.array([25., 14.]) # bounding box in cm
    states_to_bound = ['hand_px', 'hand_pz']
    neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
    drives_neurons = np.array([x in neuron_driving_states for x in state_vars])
    decoder = kfdecoder.KFDecoder(kf, mFR, sdFR, units, bounding_box, 
        state_vars, drives_neurons, states_to_bound, binlen=binlen, 
        tslice=tslice)

    # Compute sufficient stats for C and Q matrices (used to initialze CLDA)
    from clda import KFRML
    n_neurons, n_states = C.shape
    R = np.mat(np.zeros([n_states, n_states]))
    S = np.mat(np.zeros([n_neurons, n_states]))
    R_small, S_small, T, ESS = KFRML.compute_suff_stats(kin[train_inds, :], spike_counts[unit_inds,:])

    R[np.ix_(stochastic_state_inds, stochastic_state_inds)] = R_small
    S[:,stochastic_state_inds] = S_small
    
    decoder.kf.R = R
    decoder.kf.S = S
    decoder.kf.T = T
    decoder.kf.ESS = ESS
    
    return decoder

def _train_KFDecoder_visual_feedback_shuffled(*args, **kwargs):
    '''
    Train a KFDecoder from visual feedback and shuffle it
    '''
    dec = _train_KFDecoder_visual_feedback(*args, **kwargs)
    return shuffle_kf_decoder(dec)

def shuffle_kf_decoder(decoder):
    '''
    Shuffle a KFDecoder
    '''
    # generate random permutation
    import random
    inds = range(decoder.kf.C.shape[0])
    random.shuffle(inds)

    # shuffle rows of C, and rows+cols of Q
    decoder.kf.C = decoder.kf.C[inds, :]
    decoder.kf.Q = decoder.kf.Q[inds, :]
    decoder.kf.Q = decoder.kf.Q[:, inds]

    decoder.kf.C_xpose_Q_inv = decoder.kf.C.T * decoder.kf.Q.I

    # RML sufficient statistics (S and T, but not R and ESS)
    # shuffle rows of S, and rows+cols of T
    try:
        decoder.kf.S = decoder.kf.S[inds, :]
        decoder.kf.T = decoder.kf.T[inds, :]
        decoder.kf.T = decoder.kf.T[:, inds]
    except AttributeError:
        # if this decoder never had the RML sufficient statistics
        #   (R, S, T, and ESS) as attributes of decoder.kf
        pass
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
    decoder.kf.ESS = 3000.

    return decoder


def rand_KFDecoder(sim_units, state_units='cm'):
    if not state_units == 'cm': 
        raise ValueError("only works for cm right now")
    # Instantiate random seed decoder
    horiz_min, horiz_max = -14., 14.
    vert_min, vert_max = -14., 14.
    
    bounding_box = np.array([horiz_min, vert_min]), np.array([horiz_max, vert_max])
    states_to_bound = ['hand_px', 'hand_pz']

    neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
    stochastic_states = ['hand_vx', 'hand_vz']

    decoder = _train_KFDecoder_2D_sim(
        stochastic_states, neuron_driving_states, sim_units,
        bounding_box, states_to_bound, include_y=True)
    cm_to_m = 0.01
    m_to_cm = 100.
    mm_to_m = 0.001
    m_to_mm = 1000.
    decoder.kf.C *= cm_to_m
    decoder.kf.W *= m_to_cm**2
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
    neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
    states_to_bound = ['hand_px', 'hand_pz']
    states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
    drives_neurons = np.array([x in neuron_driving_states for x in states])

    args = (bounding_box, states, drives_neurons, states_to_bound)
    kwargs = dict(binlen=dt)

    # rescale beta for units
    beta[:,3:6] *= units_mult
    
    # Control input matrix for SSM for control inputs
    I = np.mat(np.eye(3))
    B = np.vstack([I, dt*1000 * I, np.zeros([1,3])])

    # instantiate Decoder
    is_stochastic = np.array([x in neuron_driving_states for x in states])
    ppf = ppfdecoder.PointProcessFilter(
            A, W, beta, dt=dt, is_stochastic=is_stochastic, B=B)

    dec = ppfdecoder.PPFDecoder(ppf, units, *args, **kwargs)

    # Force decoder to run at max 60 Hz
    dec.bmicount = 0
    dec.bminum = 0
    return dec

def _interpolate_KFDecoder_state_between_updates(decoder):
    import mpmath
    A = decoder.kf.A
    # check that dt is a multiple of 60 Hz
    power = 1./(decoder.binlen * 60)
    assert (int(power) - power) < 1e-5
    A_60hz = mpmath.powm(A, 1./(decoder.binlen * 60))
    A_60hz = np.mat(np.array(A_60hz.tolist(), dtype=np.float64))
    decoder.kf.A = A_60hz
    decoder.interpolate_using_ssm = True
    return decoder
