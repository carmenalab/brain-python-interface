import pickle
import sys

import numpy as np
from scipy.io import loadmat
from plexon import plexfile, psth
from riglib.nidaq import parse

import tables

import kfdecoder

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
    print neurons.shape
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
    
    hand_kin = kin[:, [0,kin.shape[1]/2], :]
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
    kf = kfdecoder.KalmanFilter(A, W, C[unit_inds,:], Q[np.ix_(unit_inds,unit_inds)])
    units = units[unit_inds,:]

    # instantiate KFdecoder
    bounding_box = np.array([-100., -100.]), np.array([100., 100.])
    states_to_bound = ['hand_px', 'hand_pz']
    decoder = kfdecoder.KFDecoder(kf, None, None, units, bounding_box, state_vars, states_to_bound)
    return decoder

if __name__ == '__main__':
    block = 'cart20130428_01'
    #block = 'cart20130425_05'
    files = dict(plexon='/storage/plexon/%s.plx' % block, hdf='/storage/rawdata/hdf/%s.hdf' % block)
    binlen = 0.1
    tslice = [1., 300.]
    
    decoder = _train_KFDecoder_manual_control(cells=None, binlen=0.1, tslice=[None,None],
        state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], **files) 
### Train various C, Q combinations (different state variables)
##C_xyz_pv, Q_xyz_pv = kfdecoder.KalmanFilter.MLE_obs_model(hand_kin, neurons[:,:-1])
##C_xz_pv, Q_xz_pv = kfdecoder.KalmanFilter.MLE_obs_model(hand_kin[[0,2,3,5],:], neurons[:,:-1])
##C_xz_v, Q_xz_v = kfdecoder.KalmanFilter.MLE_obs_model(hand_kin[[3,5],:], neurons[:,:-1])
##
##A_xyz_pv = _gen_A(1, 1./60, 0, 0.8, 1)
##W_xyz_pv = _gen_A(0, 0, 0, 700, 0)
##A_xz_pv = _gen_A(1, 1./60, 0, 0.8, 1, ndim=2)
##W_xz_pv = _gen_A(0, 0, 0, 700, 0, ndim=2)
##A_xz_v = A_xz_pv[2:,2:]
##W_xz_v = W_xz_pv[2:,2:]
##
##inds, = np.nonzero(np.array(C_xyz_pv)[:,-1])
##kf_xyz_pv = kfdecoder.KalmanFilter(A_xyz_pv, W_xyz_pv, C_xyz_pv[inds,:], Q_xyz_pv[np.ix_(inds, inds)])
##F_xyz_pv, K_xyz_pv = kf_xyz_pv.get_sskf()
##kf_xz_pv = kfdecoder.KalmanFilter(A_xz_pv, W_xz_pv, C_xz_pv[inds,:], Q_xz_pv[np.ix_(inds, inds)])
##F_xz_pv, K_xz_pv = kf_xz_pv.get_sskf()
##kf_xz_v = kfdecoder.KalmanFilter(A_xz_v, W_xz_v, C_xz_v[inds,:], Q_xz_v[np.ix_(inds, inds)])
##F_xz_v, K_xz_v = kf_xz_v.get_sskf()
##
### test decoder pickling
##pickle.dump(kf_xyz_pv, open('/home/helene/suraj_bmi_testing/stuff.pkl', 'w'))
##del kf_xyz_pv
##kf_xyz_pv = pickle.load(open('/home/helene/suraj_bmi_testing/stuff.pkl'))
