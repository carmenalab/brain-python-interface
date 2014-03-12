'''
Methods to create and train Decoder objects
'''
import re
import pickle
import sys

import numpy as np
from scipy.io import loadmat
from riglib.nidaq import parse

import tables
import kfdecoder, ppfdecoder
import pdb
import state_space_models
from itertools import izip

import extractor

############
## Constants
############
pi = np.pi 

empty_bounding_box = [np.array([]), np.array([])]
stoch_states_to_decode_2D_vel = ['hand_vx', 'hand_vz'] 
states_3D_endpt = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
states_explaining_neural_activity_2D_vel_decoding = ['hand_vx', 'hand_vz', 'offset']

class State(object):
    def __init__(self, name, stochastic=False, drives_obs=False, min_val=np.nan, max_val=np.nan, order=-1):
        assert not name == 'q', "'q' is a reserved keyword (symbol for generalized robot coordinates) and cannot be used as a state name"
        self.name = name
        self.stochastic = stochastic 
        self.drives_obs = drives_obs
        self.min_val = min_val
        self.max_val = max_val
        self.order = order

    def __repr__(self):
        return str(self.name) 

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        else:
            return np.all([self.__dict__[x] == other.__dict__[x] or (np.isnan(self.__dict__[x]) and np.isnan(other.__dict__[x])) for x in self.__dict__])
            # return self.__dict__

class StateSpace(object):
    def __init__(self, *states):
        self.states = list(states)

    def __repr__(self):
        return 'State space: ' + str(self.state_names)

    @property
    def is_stochastic(self):
        return np.array([x.stochastic for x in self.states])

    @property
    def drives_obs(self):
        return np.array([x.drives_obs for x in self.states])

    @property
    def state_names(self):
        return [x.name for x in self.states]

    @property
    def bounding_box(self):
        min_bounds = np.array(filter(lambda x: x is not np.nan, [x.min_val for x in self.states]))
        max_bounds = np.array(filter(lambda x: x is not np.nan, [x.max_val for x in self.states]))
        return (min_bounds, max_bounds)

    @property
    def states_to_bound(self):
        return [x.name for x in filter(lambda x: x.min_val is not np.nan, self.states)]

    @property
    def n_states(self):
        return len(self.states)

    @property
    def train_inds(self):
        return filter(lambda k: self.states[k].stochastic, range(self.n_states))

    @property
    def drives_obs_inds(self):
        return filter(lambda k: self.states[k].drives_obs, range(self.n_states))

    @property 
    def state_order(self):
        return np.array([x.order for x in self.states])

    def get_ssm_matrices(self):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, StateSpace):
            return False
        else:
            return self.states == other.states

offset_state = State('offset', stochastic=False, drives_obs=True, order=-1)

# TODO have some method of associating the A and W matrices with the State space model class
class StateSpaceEndptVel(StateSpace):
    def __init__(self):
        super(StateSpaceEndptVel, self).__init__(
            State('hand_px', stochastic=False, drives_obs=False, min_val=-25., max_val=25., order=0),
            State('hand_py', stochastic=False, drives_obs=False, order=0),
            State('hand_pz', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
            State('hand_vx', stochastic=True,  drives_obs=True, order=1),
            State('hand_vy', stochastic=False, drives_obs=False, order=1),
            State('hand_vz', stochastic=True,  drives_obs=True, order=1),
            offset_state
        )

    def get_ssm_matrices(self, update_rate=0.1):
        # State-space model set from expert data
        A, W = state_space_models.linear_kinarm_kf(update_rate=update_rate)

        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(3))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1,3])])
        return A, B, W

class StateSpaceExoArm(StateSpace):
    '''
    State space representing the kinematics of the exoskeleton
        1) shoulder flexion extension
        2) shoulder abduction/adduction
        3) elbow rotation
        4) elbow flexion/extension
        5) pronation/supination
    '''
    def __init__(self):
        super(StateSpaceExoArm, self).__init__(
                # position states
                State('sh_pflex', stochastic=False, drives_obs=False, order=0),
                State('sh_pabd', stochastic=False, drives_obs=False, order=0), 
                State('sh_prot', stochastic=False, drives_obs=False, order=0), 
                State('el_pflex', stochastic=False, drives_obs=False, order=0), 
                State('el_psup', stochastic=False, drives_obs=False, order=0), 
                # velocity states
                State('sh_vflex', stochastic=True, drives_obs=True, order=1), 
                State('sh_vabd', stochastic=True, drives_obs=True, order=1), 
                State('sh_vrot', stochastic=True, drives_obs=True, order=1), 
                State('el_vflex', stochastic=True, drives_obs=True, order=1), 
                State('el_vsup', stochastic=True, drives_obs=True, order=1), 
                # offset
                offset_state,
        )

    def get_ssm_matrices(self, update_rate=0.1):
        raise NotImplementedError("Still need to determine A for the full joint space. Need 3D reaching data from real primate")

class StateSpaceExoArm2D(StateSpaceExoArm):
    '''
    Exo arm, but limited to the 2D x-z plane by allowing only 
    should abduction/adduction and elbow flexion/extension
    '''
    def __init__(self):
        super(StateSpaceExoArm, self).__init__(
                # position states
                State('sh_pflex', stochastic=False, drives_obs=False, min_val=0, max_val=0, order=0),
                State('sh_pabd', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('sh_prot', stochastic=False, drives_obs=False, min_val=0, max_val=0, order=0),
                State('el_pflex', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('el_psup', stochastic=False, drives_obs=False, min_val=0, max_val=0, order=0),
                # velocity states
                State('sh_vflex', stochastic=False, drives_obs=False, order=1),
                State('sh_vabd', stochastic=True, drives_obs=True, order=1),
                State('sh_vrot', stochastic=False, drives_obs=False, order=1),
                State('el_vflex', stochastic=True, drives_obs=True, order=1),
                State('el_vsup', stochastic=False, drives_obs=False, order=1),
                # offset
                offset_state,
        )

    def get_ssm_matrices(self, update_rate=0.1):
        '''
        State space model from expert data
        '''
        Delta_KINARM = 1./10
        w = 0.01 #0.0007
        #w = 0.3 # TODO come up with this value more systematically!
        w_units_resc = w / 1 # velocity will always be in radians/sec
        a_resampled, w_resampled = state_space_models.resample_scalar_ssm(0.8, w_units_resc, Delta_old=Delta_KINARM, Delta_new=update_rate)

        # TODO get the number of dimensions from the arm configuration (i.e. a method to return the order of each state
        ndim = 5 # NOTE: This is the number of 1st order states, not the dimension of the state vector
        A = state_space_models._gen_A(1, update_rate, 0, a_resampled, 1, ndim=ndim)
        W = state_space_models._gen_A(0, 0, 0, w_resampled, 0, ndim=ndim)
        
        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(ndim))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1, ndim])])
        return A, B, W

class StateSpaceFourLinkTentacle2D(StateSpace):
    def __init__(self):
        super(StateSpaceFourLinkTentacle2D, self).__init__(
                # position states
                State('sh_pabd', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('el_pflex', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('wr_pflex', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('fi_pflex', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                # velocity states
                State('sh_vabd', stochastic=True, drives_obs=True, order=1),
                State('el_vflex', stochastic=True, drives_obs=True, order=1),
                State('wr_vflex', stochastic=True, drives_obs=True, order=1),
                State('fi_vflex', stochastic=True, drives_obs=True, order=1),
                # offset
                offset_state,
        )

    def get_ssm_matrices(self, update_rate=0.1):
        '''
        State space model from expert data
        '''
        Delta_KINARM = 1./10
        w = 0.01 #0.0007
        #w = 0.3 # TODO come up with this value more systematically!
        w_units_resc = w / 1 # velocity will always be in radians/sec
        a_resampled, w_resampled = state_space_models.resample_scalar_ssm(0.8, w_units_resc, Delta_old=Delta_KINARM, Delta_new=update_rate)

        # TODO get the number of dimensions from the arm configuration (i.e. a method to return the order of each state
        ndim = 4 # NOTE: This is the number of 1st order states, not the dimension of the state vector
        A = state_space_models._gen_A(1, update_rate, 0, a_resampled, 1, ndim=ndim)
        W = state_space_models._gen_A(0, 0, 0, w_resampled, 0, ndim=ndim)
        
        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(ndim))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1, ndim])])
        return A, B, W        

endpt_2D_state_space = StateSpaceEndptVel()
joint_2D_state_space = StateSpaceExoArm2D()
tentacle_2D_state_space = StateSpaceFourLinkTentacle2D()

################################################
## Functions to train endpoint velocity decoders
################################################
def train_endpt_velocity_PPFDecoder(kin, spike_counts, units, update_rate=0.1, tslice=None, _ssm=endpt_2D_state_space):
    '''
    Train a Point-process filter decoder which predicts the endpoint velocity
    '''
    binlen = update_rate
    n_features = spike_counts.shape[0]  # number of neural features

    # C should be trained on all of the stochastic state variables, excluding 
    # the offset terms
    C = np.zeros([n_features, _ssm.n_states])
    C[:, _ssm.drives_obs_inds], pvals = ppfdecoder.PointProcessFilter.MLE_obs_model(kin[_ssm.train_inds, :], spike_counts)
    
    # Set state space model
    A, B, W = _ssm.get_ssm_matrices(update_rate=update_rate)

    # instantiate Decoder
    ppf = ppfdecoder.PointProcessFilter(
            A, W, C, B=B, dt=update_rate, is_stochastic=_ssm.is_stochastic)
    decoder = ppfdecoder.PPFDecoder(ppf, units, _ssm.bounding_box, 
        _ssm.state_names, _ssm.drives_obs, _ssm.states_to_bound, binlen=binlen, 
        tslice=tslice)

    decoder.ssm = _ssm

    decoder.n_features = n_features
    
    return decoder

def train_KFDecoder(_ssm, kin, neural_features, units, update_rate=0.1, tslice=None):
    binlen = update_rate
    n_features = neural_features.shape[0]  # number of neural features

    # C should be trained on all of the stochastic state variables, excluding the offset terms
    C = np.zeros([n_features, _ssm.n_states])
    C[:, _ssm.drives_obs_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(kin[_ssm.train_inds, :], neural_features)

    mFR = np.mean(neural_features, axis=1)
    sdFR = np.std(neural_features, axis=1)

    # Set state space model
    A, B, W = _ssm.get_ssm_matrices(update_rate=update_rate)

    # instantiate KFdecoder
    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=_ssm.is_stochastic)
    decoder = kfdecoder.KFDecoder(kf, mFR, sdFR, units, _ssm.bounding_box, 
        _ssm.state_names, _ssm.drives_obs, _ssm.states_to_bound, binlen=binlen, 
        tslice=tslice)

    # Compute sufficient stats for C and Q matrices (used for RML CLDA)
    from clda import KFRML
    n_features, n_states = C.shape
    R = np.mat(np.zeros([n_states, n_states]))
    S = np.mat(np.zeros([n_features, n_states]))
    R_small, S_small, T, ESS = KFRML.compute_suff_stats(kin[_ssm.train_inds, :], neural_features)

    R[np.ix_(_ssm.drives_obs_inds, _ssm.drives_obs_inds)] = R_small
    S[:,_ssm.drives_obs_inds] = S_small
    
    decoder.kf.R = R
    decoder.kf.S = S
    decoder.kf.T = T
    decoder.kf.ESS = ESS
    
    decoder.ssm = _ssm

    decoder.n_features = n_features

    return decoder

###################
## Helper functions
###################
def unit_conv(starting_unit, ending_unit):
    ''' Convert between units, e.g. cm to m
    Lookup table for conversion factors between units; this function exists
    only to avoid hard-coded constants in most of the code
    '''

    if starting_unit == ending_unit:
        return 1
    elif (starting_unit, ending_unit) == ('cm', 'm'):
        return 0.01
    elif (starting_unit, ending_unit) == ('m', 'cm'):
        return 100

def obj_eq(self, other, attrs=[]):
    ''' Determine if two objects have mattching array attributes
    '''
    if isinstance(other, type(self)):
        attrs_eq = filter(lambda y: y in other.__dict__, filter(lambda x: x in self.__dict__, attrs))
        equal = map(lambda attr: np.array_equal(getattr(self, attr), getattr(other, attr)), attrs_eq)
        return np.all(equal)
    else:
        return False
    
def obj_diff(self, other, attrs=[]):
    ''' Calculate the difference of the two objects w.r.t the specified attributes
    '''
    if isinstance(other, type(self)):
        attrs_eq = filter(lambda y: y in other.__dict__, filter(lambda x: x in self.__dict__, attrs))
        diff = map(lambda attr: getattr(self, attr) - getattr(other, attr), attrs_eq)
        return np.array(diff)
    else:
        return False
    
def lookup_cells(cells):
    ''' Convert string names of units to 'machine' format.
    Take a list of neural units specified as a list of strings and convert 
    to the 2D array format used to specify neural units to train decoders
    '''
    cellname = re.compile(r'(\d{1,3})\s*(\w{1})')
    cells = [ (int(c), ord(u) - 96) for c, u in cellname.findall(cells)]
    return cells

def inflate(A, current_states, full_state_ls, axis=0):
    ''' 'Inflate' a matrix by filling in rows/columns with zeros '''
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

def sys_eq(sys1, sys2):
    return sys1 in [sys2, sys2[1:]]

#####################
## Data Preprocessing
#####################
def _get_tmask(plx, tslice, syskey_fn=lambda x: x[0] in ['task', 'ask'], sys_name='task'):
    ''' Find the rows of the plx file to use for training the decoder
    '''
    # Open plx file
    from plexon import plexfile
    if isinstance(plx, str) or isinstance(plx, unicode):
        plx = plexfile.openFile(plx)
    events = plx.events[:].data
    syskey=0

    # get system registrations
    reg = parse.registrations(events)

    # find the key for the motiontracker system data
    for key, system in reg.items():
        if sys_eq(system[0], sys_name):
        #if syskey_fn(system):
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

def get_spike_counts(plx, neurows, binlen, units, extractor_kwargs):
    '''Compute binned spike count features
    '''

    # interpolate between the rows to 180 Hz
    if binlen < 1./60:
        interp_rows = []
        neurows = np.hstack([neurows[0] - 1./60, neurows])
        for r1, r2 in izip(neurows[:-1], neurows[1:]):
            interp_rows += list(np.linspace(r1, r2, 4)[1:])
        interp_rows = np.array(interp_rows)
    else:
        step = int(binlen/(1./60)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
        interp_rows = neurows[::step]

    from plexon import psth
    spike_bin_fn = psth.SpikeBin(units, binlen)
    spike_counts = np.array(list(plx.spikes.bin(interp_rows, spike_bin_fn)))

    # discard units that never fired at all
    # TODO -- verify that summing over axis 0 is correct
    unit_inds, = np.nonzero(np.sum(spike_counts, axis=0))
    units = units[unit_inds,:]
    spike_counts = spike_counts[:, unit_inds]
    extractor_kwargs['units'] = units

    return spike_counts, units, extractor_kwargs

def get_lfp_power(plx, neurows, binlen, units, extractor_kwargs):
    '''Compute lfp power features
    '''
    
    # interpolate between the rows to 180 Hz
    if binlen < 1./60:
        interp_rows = []
        neurows = np.hstack([neurows[0] - 1./60, neurows])
        for r1, r2 in izip(neurows[:-1], neurows[1:]):
            interp_rows += list(np.linspace(r1, r2, 4)[1:])
        interp_rows = np.array(interp_rows)
    else:
        step = int(binlen/(1./60)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
        interp_rows = neurows[::step]

    fs = 1000.
    win_len     = extractor_kwargs['win_len']
    bands       = extractor_kwargs['bands']
    channels    = extractor_kwargs['channels']
    filt_order  = extractor_kwargs['filt_order']

    from scipy.signal import butter, lfilter

    filt_coeffs = dict()
    for band in bands:
        nyq = 0.5 * fs
        low = band[0] / nyq
        high = band[1] / nyq
        filt_coeffs[band] = butter(filt_order, [low, high], btype='band')  # returns (b, a)

    n_itrs = len(interp_rows)
    n_chan = len(channels)

    n_pts = int(win_len * fs)
        
    lfp_power = np.zeros((n_itrs, n_chan * len(bands)))
    for i, t in enumerate(interp_rows):
        for j, band in enumerate(bands):
            b, a = filt_coeffs[band]
            y = lfilter(b, a, plx.lfp[t-win_len:t].data[:, channels-1])
            lfp_power[i, j*n_chan:(j+1)*n_chan] = np.log10((1. / n_pts) * np.sum(y**2, axis=0))
    
    # TODO -- discard any channel(s) for which the log power in any frequency 
    #   bands was ever equal to -inf (i.e., power was equal to 0)
    # or, perhaps just add a small epsilon inside the log to avoid this
    # then, remember to do this:  extractor_kwargs['channels'] = channels

    return lfp_power, units, extractor_kwargs

def get_cursor_kinematics(hdf, binlen, tmask, update_rate_hz=60., key='cursor'):
    ''' Get positions and calculate velocity

    Note: the two different cases below appear to calculate the velocity in two 
    different ways. This is purely for legacy reasons, i.e. the second method
    is intentionally slightly different from the first.
    '''
    kin = hdf.root.task[:][key]

    ##### this is to test on files that didn't save the full 5-joint kinematics, remove soon!
    if key=='joint_angles' and kin.shape[1]==2:
        newja = np.zeros([len(kin), 5])
        newja[:,1] = kin[:,1]
        newja[:,3] = kin[:,0]
        kin = newja
    ##########################

    inds, = np.nonzero(tmask)
    step_fl = binlen/(1./update_rate_hz)
    if step_fl < 1: # more than one spike bin per kinematic obs
        velocity = np.diff(kin, axis=0) * update_rate_hz
        velocity = np.vstack([np.zeros(kin.shape[1]), velocity])
        kin = np.hstack([kin, velocity])

        n_repeats = int((1./update_rate_hz)/binlen)
        inds = np.sort(np.hstack([inds]*n_repeats))
        kin = kin[inds]
    else:
        step = int(binlen/(1./update_rate_hz))
        inds = inds[::step]
        kin = kin[inds]

        velocity = np.diff(kin, axis=0) * 1./binlen
        velocity = np.vstack([np.zeros(kin.shape[1]), velocity])
        kin = np.hstack([kin, velocity])

    return kin

def get_joint_kinematics(cursor_kin, shoulder_center, binlen=0.1):
    '''
    Use inverse kinematics to calculate the joint angles corresponding to 
    a particular endpoint trajectory. Note that this is only unique for 2D 
    planar movements; General 3D movements will require a different method, 
    still to be implemented

    NOTE: the binlength would not be required if the joint velocities were
    calculated from the endpoint velocities using the Jacobian transformation!
    (see note below
    '''
    from riglib.stereo_opengl import ik
    endpoint_pos = cursor_kin[:,0:3] + shoulder_center
    
    # Calculate joint angles using the IK methods
    # TODO arm link lengths are hard coded right now!
    joint_angles = ik.inv_kin_2D(endpoint_pos, 20., 15.)
    joint_angles = np.vstack(joint_angles[x] for x in joint_angles.dtype.names).T
    #joint_angles_2D = np.vstack([joint_angles['sh_pabd'], joint_angles['el_pflex']]).T
    
    # get joint velocities; TODO: use pointwise diff or jacobian?
    joint_vel_2D = np.diff(joint_angles, axis=0) * 1./binlen
    joint_vel_2D = np.vstack([np.zeros(5), joint_vel_2D])
    joint_kin = np.hstack([joint_angles, joint_vel_2D])
    return joint_kin

def preprocess_files(files, binlen, units, tslice, extractor_cls, extractor_kwargs, source='task', kin_var='cursor'):
    plx_fname = str(files['plexon']) 
    from plexon import plexfile
    try:
        plx = plexfile.openFile(plx_fname)
    except IOError:
        print "Could not open .plx file: %s" % plx_fname
        raise Exception
    
    # Use all of the units if none are specified
    if units == None:
        units = np.array(plx.units).astype(np.int32)

    tmask, rows = _get_tmask(plx, tslice, syskey_fn=lambda x: x[0] in [source, source[1:]])
    
    hdf = tables.openFile(files['hdf'])
    kin = get_cursor_kinematics(hdf, binlen, tmask, key=kin_var)
    neurows = rows[tmask]

    # TODO -- make the get_spike_counts and get_lfp_power functions part of their respective classes
    if extractor_cls == extractor.BinnedSpikeCountsExtractor:
        neural_features, units, extractor_kwargs = get_spike_counts(plx, neurows, binlen, units, extractor_kwargs)
    elif extractor_cls == extractor.LFPButterBPFPowerExtractor:
        neural_features, units, extractor_kwargs = get_lfp_power(plx, neurows, binlen, units, extractor_kwargs)
    else:
        raise Exception("Unrecognized feature type!")

    return kin, neural_features, units, extractor_kwargs


def _train_PPFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=None, binlen=1./180, tslice=[None,None], 
    _ssm=endpt_2D_state_space, source='task', kin_var='cursor', shuffle=False, **files):
    '''
    Train a PPFDecoder from visual feedback
    '''
    binlen = 1./180 # TODO remove hardcoding!

    kin, spike_counts, units, extractor_kwargs = preprocess_files(files, binlen, units, tslice, extractor_cls, extractor_kwargs, source=source, kin_var=kin_var)
    if len(kin) != len(spike_counts):
        raise ValueError('Training data and neural data are the wrong length: %d vs. %d'%(len(kin), len(spike_counts)))
    
    # Remove 1st kinematic sample and last spike counts sample to align the 
    # velocity with the spike counts
    kin = kin[1:].T
    spike_counts = spike_counts[:-1].T

    spike_counts[spike_counts > 1] = 1
    decoder = train_endpt_velocity_PPFDecoder(kin, spike_counts, units, update_rate=binlen, tslice=tslice, _ssm=_ssm)

    # save extractor info into the decoder so we can create the appropriate
    #   extractor object when we use this decoder later on
    decoder.extractor_cls = extractor_cls
    decoder.extractor_kwargs = extractor_kwargs

    if shuffle: decoder.shuffle()

    return decoder

def _train_KFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=None, binlen=0.1, tslice=[None,None], 
    _ssm=None, source='task', kin_var='cursor', shuffle=False, **files):
    '''
    Train a KFDecoder from visual feedback
    '''
    kin, neural_features, units, extractor_kwargs = preprocess_files(files, binlen, units, tslice, extractor_cls, extractor_kwargs, source=source, kin_var=kin_var)
    if _ssm == None:
        if kin_var == 'cursor':
            _ssm=endpt_2D_state_space
        elif kin_var == 'joint_angles':
            _ssm=joint_2D_state_space
        
    if len(kin) != len(neural_features):
        raise ValueError('Training data and neural data are the wrong length: %d vs. %d'%(len(kin), len(neural_features)))
    
    # Remove 1st kinematic sample and last neural features sample to align the 
    # velocity with the neural features
    kin = kin[1:].T
    neural_features = neural_features[:-1].T

    decoder = train_KFDecoder(_ssm, kin, neural_features, units, update_rate=binlen, tslice=tslice)

    # save extractor info into the decoder so we can create the appropriate
    #   extractor object when we use this decoder later on
    decoder.extractor_cls = extractor_cls
    decoder.extractor_kwargs = extractor_kwargs

    if shuffle: decoder.shuffle()

    return decoder

def _train_joint_KFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=None, binlen=0.1, tslice=[None,None],
    _ssm=joint_2D_state_space, source='task', kin_var='joint_angles', shuffle=False, **files):
    '''
    One-liner to train a 2D2L joint BMI. To be removed as soon as the train BMI gui can be updated
    to have more arguments
    '''
    return _train_KFDecoder_visual_feedback(units=units, binlen=binlen, tslice=tslice,
                                            _ssm=_ssm, source=source, kin_var=kin_var,
                                            shuffle=shuffle, **files)

def _train_tentacle_KFDecoder_visual_feedback(extractor_cls, extractor_kwargs, units=None, binlen=0.1, tslice=[None, None], 
    _ssm=tentacle_2D_state_space, source='task', kin_var='joint_angles', shuffle=False, **files):
    '''
    One-liner to train a tentacle BMI. To be removed as soon as the train BMI gui can be updated
    to have more arguments
    '''
    print "using tentacletate space " , _ssm == tentacle_2D_state_space
    return _train_KFDecoder_visual_feedback(units=units, binlen=binlen, tslice=tslice, 
                                            _ssm=_ssm, source=source, kin_var=kin_var, 
                                            shuffle=shuffle, **files)

def _train_KFDecoder_cursor_epochs(extractor_cls, extractor_kwargs, units=None, binlen=0.1, tslice=[None,None], 
    state_vars=['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], 
    exclude_targ_ind=[-1, 0], **files):
    '''
    Train a KFDecoder from cursor movement on screen, but only for selected epochs
    Define which epochs you want to remove from training: 
        exclude_targ_ind=[-1, 0] removes original state and when origin is displayed
    '''
    update_rate=binlen
    # Open plx file
    plx_fname = str(files['plexon'])
    from plexon import plexfile
    plx = plexfile.openFile(plx_fname)

    # Compute last spike and set it to tslice[1]: 
    last_spk = plx.spikes[:].data[-1][0]

    if tslice[1]==None:
        tslice=[0,int(last_spk)]
    else:
        print tslice
        print tslice[1]
        print int(last_spk)
        tslice=[tslice[0], np.min([tslice[1],int(last_spk)])]

    tmask, rows = _get_tmask(plx_fname, tslice, syskey_fn=lambda x: x[0] in ['task', 'ask'])
    tmask_continuous = np.array_equal(np.unique(np.diff(np.nonzero(tmask)[0])), np.array([1]))

    #Grab masked kinematic data
    h5 = tables.openFile(files['hdf'])
    kin = h5.root.task[tmask]['cursor']
    targ_ind = h5.root.task[tmask]['target_index']
    step = int(binlen/(1./60)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
    kin = kin[::step, :]
    velocity = np.diff(kin, axis=0) * 1./binlen
    kin = kin[1:] #Remove 1st element to match size of velocity

    ##Select the epochs to use. sub_targ_ind returns indices of kin, spike_counts where targ_ind is desired value 
    tmp = targ_ind != exclude_targ_ind
    targ_ind = np.sum(tmp,axis=1) == len(exclude_targ_ind)
    sub_targ_ind = np.zeros([len(kin)])
    for i in xrange(len(sub_targ_ind)):
        sub_targ_ind[i] = sum(targ_ind[i*step:((i+1)*step)])==step
    sub_targ_ind = np.where(sub_targ_ind>0)

    # Use all of the units if none are specified
    if units == None:
        units = np.array(plx.units).astype(np.int32)

    ## Bin the neural data
    from plexon import psth
    spike_bin_fn = psth.SpikeBin(units, binlen)
    neurows = rows[tmask]
    neurows = neurows[::step]
    spike_counts = np.array(list(plx.spikes.bin(neurows, spike_bin_fn)))

    ##Index spike_counts, kin, velocity with selected epochs 
    spike_counts = spike_counts[:-1,:] #Remove last count to match size of velocity
    
    spike_counts = spike_counts[sub_targ_ind,:][0]
    kin = kin[sub_targ_ind, :][0]
    velocity = velocity[sub_targ_ind,:][0]

    ##Only use some of spike counts
    if len(kin) != len(spike_counts):
        raise ValueError('Training data and neural data are the wrong length: %d vs. %d'%(len(kin), len(spike_counts)))
    
    kin = np.hstack([kin, velocity]).T
    spike_counts = spike_counts.T

    decoder = train_KFDecoder(endpt_2D_state_space, kin, spike_counts, units, update_rate=binlen, tslice=tslice)

    # save extractor info into the decoder so we can create the appropriate
    #   extractor object when we use this decoder later on
    extractor_kwargs['units'] = units
    decoder.extractor_cls = extractor_cls
    decoder.extractor_kwargs = extractor_kwargs

    return decoder


def _train_KFDecoder_manual_control(extractor_cls, extractor_kwargs, units=None, binlen=0.1, tslice=[None,None], 
    state_vars=['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset'], 
    stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    #state_vars=['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset'], 
    #stochastic_vars=['hand_vx', 'hand_vz', 'offset'], **files):
    """
    Train KFDecoder from manual control
    """
    # Open plx file
    from plexon import plexfile
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
    
    # Use all of the units if none are specified
    if units == None:
        units = np.array(plx.units).astype(np.int32)

    # Create PSTH function
    from plexon import psth
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
    
    # save extractor info into the decoder so we can create the appropriate
    #   extractor object when we use this decoder later on
    extractor_kwargs['units'] = units
    decoder.extractor_cls = extractor_cls
    decoder.extractor_kwargs = extractor_kwargs

    return decoder

#######################
## Simulation functions
#######################
def _train_PPFDecoder_2D_sim(stochastic_states, neuron_driving_states, units,
    bounding_box, states_to_bound, include_y=True, dt=0.1, v=0.4):
    '''
    Train a simulation PPFDecoder
    '''
    raise NotImplementedError

def _train_KFDecoder_2D_sim(_ssm, units, dt=0.1):
    n_neurons = units.shape[0]
    ###if include_y:
    ###    states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
    ###    A = np.array([[1, 0, 0, dt, 0,  0,  0],
    ###                  [0, 1, 0, 0,  dt, 0,  0],
    ###                  [0, 0, 1, 0,  0,  dt, 0],
    ###                  [0, 0, 0, v,  0,  0,  0],
    ###                  [0, 0, 0, 0,  v,  0,  0],
    ###                  [0, 0, 0, 0,  0,  v,  0],
    ###                  [0, 0, 0, 0,  0,  0,  1]])
    ###else:
    ###    states = ['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset']
    ###    A = np.array([[1, 0, dt, 0, 0],
    ###                  [0, 1, 0, dt, 0],
    ###                  [0, 0, v,  0, 0],
    ###                  [0, 0, 0,  v, 0],
    ###                  [0, 0, 0,  0, 1]])

    states = _ssm.state_names
    A, B, W = _ssm.get_ssm_matrices(update_rate=dt)
    drives_neurons = _ssm.drives_obs
    is_stochastic = _ssm.is_stochastic
    bounding_box = _ssm.bounding_box
    states_to_bound = _ssm.states_to_bound
    nX = _ssm.n_states

    C = np.random.standard_normal([n_neurons, nX])
    C[:, ~drives_neurons] = 0
    Q = 10 * np.identity(n_neurons) 

    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=is_stochastic)

    mFR = 0
    sdFR = 1
    decoder = kfdecoder.KFDecoder(kf, mFR, sdFR, units, bounding_box, 
        states, drives_neurons, states_to_bound)

    cm_to_m = 0.01
    decoder.kf.R = np.mat(np.identity(decoder.kf.C.shape[1]))
    decoder.kf.S = decoder.kf.C * cm_to_m
    decoder.kf.T = decoder.kf.Q + decoder.kf.S*decoder.kf.S.T
    decoder.kf.ESS = 3000.

    cm_to_m = 0.01
    m_to_cm = 100.
    mm_to_m = 0.001
    m_to_mm = 1000.
    decoder.kf.C *= cm_to_m

    decoder.ssm = _ssm
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
    """
    Create KFDecoder from MATLAB decoder file used in a Dexterit-based
    BMI
    """
    decoder_data = loadmat(decoder_fname)['decoder']
    A = decoder_data['A'][0,0]
    W = decoder_data['W'][0,0]
    H = decoder_data['H'][0,0]
    Q = decoder_data['Q'][0,0]
    mFR = decoder_data['mFR'][0,0].ravel()
    sdFR = decoder_data['sdFR'][0,0].ravel()

    pred_sigs = [str(x[0]) for x in decoder_data['predSig'][0,0].ravel()]
    unit_lut = {'a':1, 'b':2, 'c':3, 'd':4}
    units = [(int(sig[3:6]), unit_lut[sig[-1]]) for sig in pred_sigs]

    drives_neurons = np.array([False, False, True, True, True])

    kf = kfdecoder.KalmanFilter(A, W, H, Q)
    dec = kfdecoder.KFDecoder(kf, mFR, sdFR, units, bounding_box, states, drives_neurons, states_to_bound)

    # Load bounder for position state
    from state_bounders import RectangularBounder
    bounding_box_data = loadmat('/Users/sgowda/bmi/workspace/decoder_switching/jeev_center_out_bmi_targets.mat')
    center_pos = bounding_box_data['centerPos'].ravel()
    px_min, py_min = center_pos - 0.09
    px_max, py_max = center_pos + 0.09
    bounding_box = [(px_min, px_max), (py_min, py_max)]
    bounder = RectangularBounder([px_min, py_min], [px_max, py_max], ['p_x', 'p_y'])    
    dec.bounder = bounder

    return dec

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
    beta = inflate(beta, neuron_driving_states, states, axis=1)

    args = (bounding_box, states, drives_neurons, states_to_bound)
    kwargs = dict(binlen=dt)

    # rescale beta for units
    beta[:,3:6] *= units_mult
    
    # Control input matrix for SSM for control inputs
    I = np.mat(np.eye(3))
    B = np.vstack([I, dt*1000 * I, np.zeros([1,3])])

    # instantiate Decoder
    stochastic_states = ['hand_vx', 'hand_vz']
    is_stochastic = np.array([x in stochastic_states for x in states])
    ppf = ppfdecoder.PointProcessFilter(
            A, W, beta, dt=dt, is_stochastic=is_stochastic, B=B)

    dec = ppfdecoder.PPFDecoder(ppf, units, *args, **kwargs)

    # Force decoder to run at max 60 Hz
    dec.bminum = 0
    return dec

def load_PPFDecoder_from_mat_file(fname, state_units='cm'):
    data = loadmat(fname)
    a = data['A'][2,2]
    w = data['W'][0,0]

    if 'T_loop' in data:
        dt = data['T_loop'][0,0]
    else:
        dt = 0.005

    spike_rate_dt = 0.001 # This is hardcoded b/c the value in the MATLAB file is probably wrong.
    A = state_space_models._gen_A(1, dt, 0, a, 1, ndim=3)
    W = state_space_models._gen_A(0, 0, 0, w, 0, ndim=3)

    if 'beta_hat' in data:
        beta = data['beta_hat'][:,:,0]
    else:
        beta = data['beta']

    beta = ppfdecoder.PointProcessFilter.frommlab(beta)
    beta[:,:-1] /= unit_conv('m', state_units)
    beta_full = inflate(beta, states_explaining_neural_activity_2D_vel_decoding, states_3D_endpt, axis=1)
    
    states = states_3D_endpt#['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
    neuron_driving_states = states_explaining_neural_activity_2D_vel_decoding#['hand_vx', 'hand_vz', 'offset'] 
    ## beta_full = inflate(beta, neuron_driving_states, states, axis=1)

    stochastic_states = ['hand_vx', 'hand_vz']  
    is_stochastic = map(lambda x: x in stochastic_states, states)

    unit_names = [str(x[0]) for x in data['decoder']['predSig'][0,0][0]]
    units = [(int(x[3:6]), ord(x[-1]) - (ord('a') - 1)) for x in unit_names]
    units = np.vstack(units)

    ppf = ppfdecoder.PointProcessFilter(A, W, beta_full, dt, is_stochastic=is_stochastic)
    ppf.spike_rate_dt = spike_rate_dt

    drives_neurons = np.array([x in neuron_driving_states for x in states])
    dec = ppfdecoder.PPFDecoder(ppf, units, empty_bounding_box, states, drives_neurons, [], binlen=dt)

    if state_units == 'cm':
        dec.filt.W[3:6, 3:6] *= unit_conv('m', state_units)**2
    return dec 

###################
## To be deprecated
###################
def _train_PPFDecoder_visual_feedback_shuffled(*args, **kwargs):
    decoder = _train_PPFDecoder_visual_feedback(*args, **kwargs)
    import random
    inds = range(decoder.filt.C.shape[0])
    random.shuffle(inds)

    # shuffle rows of C, and rows+cols of Q
    decoder.filt.C = decoder.filt.C[inds, :]
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

