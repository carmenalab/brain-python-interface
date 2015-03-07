'''
Methods to create and train Decoders
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

import stat
import os
import subprocess

############
## Constants
############
pi = np.pi 

################################################################################
## Functions to synchronize task-generated HDF files and neural recording files
################################################################################
def sys_eq(sys1, sys2):
    '''
    Determine if two system strings match. A separate function is required because sometimes
    the NIDAQ card doesn't properly transmit the first character of the name of the system.

    Parameters
    ----------
    sys1: string
        Name of system from the neural file
    sys2: string
        Name of system to match

    Returns
    -------
    Boolean indicating whether sys1 and sys2 match
    '''
    return sys1 in [sys2, sys2[1:]]


FAKE_BLACKROCK_TMASK = True
# FAKE_BLACKROCK_TMASK = False

########################################
## Neural data synchronization functions
########################################
def _get_tmask(files, tslice, sys_name='task'):
    if 'plexon' in files:
        fn = _get_tmask_plexon
        fname = str(files['plexon'])
    elif 'blackrock' in files:
        if FAKE_BLACKROCK_TMASK:
            fn = _get_tmask_blackrock_fake
            fname = files['hdf']
        else:
            fn = _get_tmask_blackrock
            fname = [str(name) for name in files['blackrock'] if '.nev' in name][0]  # only one of them
    else:
        raise Exception("Neural data file(s) not found!")

    return fn(fname, tslice, sys_name=sys_name)

def _get_tmask_plexon(plx, tslice, sys_name='task'):
    '''
    Find the rows of the plx file to use for training the decoder

    Parameters
    ----------
    plx : plexfile instance
        The plexon file to sync
    tslice : list of length 2
        Specify the start and end time to examine the file, in seconds
    sys_name : string, optional
        The "system" being synchronized. When the task is running, each data source 
        (i.e., each HDF table) is allowed to be asynchronous and thus is independently 
        synchronized with the neural recording system.

    Returns
    -------
    tmask: np.ndarray of shape (N, ) of booleans
        Specifies which entries of "rows" (see below) are within the time bounds
    rows: np.ndarray of shape (N, ) of integers
        The times at which rows of the specified HDF table were recieved in the neural recording box
    '''
    # Open plx file
    from plexon import plexfile
    if isinstance(plx, str) or isinstance(plx, unicode):
        plx = plexfile.openFile(plx)

    # Get the list of all the systems registered in the neural data file
    events = plx.events[:].data
    reg = parse.registrations(events)

    if len(reg.keys()) > 0:
        # find the key for the specified system data
        syskey = None
        for key, system in reg.items():
            if sys_eq(system[0], sys_name):
                syskey = key
                break

        if syskey is None:
            raise Exception('riglib.bmi.train._get_tmask: Training data source not found in neural data file!')        
    elif len(reg.keys()) == 0:
        # try to find how many systems' rowbytes were in the HDF file
        rowbyte_data = parse.rowbyte(events)
        if len(rowbyte_data.keys()) == 1:
            syskey = rowbyte_data.keys()[0]
        else:
            raise Exception("No systems registered and I don't know which sys to use to train!")

    # get the corresponding hdf rows
    rows = parse.rowbyte(events)[syskey][:,0]
    
    # Determine which rows are within the time bounds
    lower, upper = 0 < rows, rows < rows.max() + 1
    l, u = tslice
    if l is not None:
        lower = l < rows
    if u is not None:
        upper = rows < u
    tmask = np.logical_and(lower, upper)
    return tmask, rows

def _get_tmask_blackrock(nev_fname, tslice, sys_name='task'):
    ''' Find the rows of the nev file to use for training the decoder.'''

    nev_hdf_fname = nev_fname + '.hdf'
    if not os.path.isfile(nev_hdf_fname):
        # convert .nev file to hdf file using Blackrock's n2h5 utility
        subprocess.call(['n2h5', nev_fname, nev_hdf_fname])

    import h5py
    nev_hdf = h5py.File(nev_hdf_fname, 'r')

    path = 'channel/digital00001/digital_set'
    ts = nev_hdf.get(path).value['TimeStamp']
    msgs = nev_hdf.get(path).value['Value']

    # copied from riglib/nidaq/parse.py
    msgtype_mask = 0b0000111<<8
    auxdata_mask = 0b1111000<<8
    rawdata_mask = 0b11111111
    msgtype = np.right_shift(np.bitwise_and(msgs, msgtype_mask), 8).astype(np.uint8)
    # auxdata = np.right_shift(np.bitwise_and(msgs, auxdata_mask), 8).astype(np.uint8)
    auxdata = np.right_shift(np.bitwise_and(msgs, auxdata_mask), 8+3).astype(np.uint8)
    rawdata = np.bitwise_and(msgs, rawdata_mask)

    # data is an N x 4 matrix that will be the argument to parse.registrations()
    data = np.vstack([ts, msgtype, auxdata, rawdata]).T

    # get system registrations
    reg = parse.registrations(data)

    syskey = None

    for key, system in reg.items():
        if sys_eq(system[0], sys_name):
            syskey = key
            break

    if syskey is None:
        raise Exception('No source registration saved in the file!')

    # get the corresponding hdf rows
    rows = parse.rowbyte(data)[syskey][:,0]

    rows = rows / 30000.
    
    lower, upper = 0 < rows, rows < rows.max() + 1
    l, u = tslice
    if l is not None:
        lower = l < rows
    if u is not None:
        upper = rows < u
    tmask = np.logical_and(lower, upper)

    return tmask, rows

def _get_tmask_blackrock_fake(hdf_fname, tslice, **kwargs):
    # need to create fake "rows" and "tmask" variables

    print 'WARNING: Using _get_tmask_blackrock_fake function!!'
    
    binlen = 0.1
    strobe_rate = 10
    hdf = tables.openFile(hdf_fname)

    n_rows = hdf.root.task[:]['plant_pos'].shape[0]
    first_ts = binlen
    rows = np.linspace(first_ts, first_ts + (n_rows-1)*(1./strobe_rate), num=n_rows)
    lower, upper = 0 < rows, rows < rows.max() + 1
    l, u = tslice
    if l is not None:
        lower = l < rows
    if u is not None:
        upper = rows < u
    tmask = np.logical_and(lower, upper)

    return tmask, rows
    
################################################################################
## Feature extraction
################################################################################
def _get_neural_features_plx(files, binlen, extractor_fn, extractor_kwargs, tslice=None, units=None, source='task', strobe_rate=60.):
    '''
    Extract the neural features used to train the decoder

    Parameters
    ----------
    files: dict
        keys of the dictionary are file-types (e.g., hdf, plexon, etc.), values are file names
    binlen: float
        Specifies the temporal resolution of the feature extraction
    extractor_fn: callable
        Function must have the call signature 
        neural_features, units, extractor_kwargs = extractor_fn(plx, neurows, binlen, units, extractor_kwargs)
    extractor_kwargs: dictionary
        Additional keyword arguments to the extractor_fn (specific to each feature extractor)

    Returns
    -------
    neural_features: np.ndarrya of shape (n_features, n_timepoints)
        Values of each feature to be used in training the decoder parameters
    units: np.ndarray of shape (N, -1)
        Specifies identty of each neural feature
    extractor_kwargs: dictionary
        Keyword arguments used to construct the feature extractor used online
    '''
    
    hdf = tables.openFile(files['hdf'])

    plx_fname = str(files['plexon']) 
    from plexon import plexfile
    try:
        plx = plexfile.openFile(plx_fname)
    except IOError:
        raise Exception("Could not open .plx file: %s" % plx_fname)
    
    # Use all of the units if none are specified
    if units == None:
        units = np.array(plx.units).astype(np.int32)

    tmask, rows = _get_tmask_plexon(plx, tslice, sys_name=source)
    neurows = rows[tmask]

    neural_features, units, extractor_kwargs = extractor_fn(files, neurows, binlen, units, extractor_kwargs)

    return neural_features, units, extractor_kwargs

def _get_neural_features_blackrock(files, binlen, extractor_fn, extractor_kwargs, tslice=None, units=None, source='task', strobe_rate=10.):    
    if units == None:
        raise Exception('"units" variable is None in preprocess_files!')

    # Note: blackrock units are actually 0-based, but the units to be used for training
    #       (which comes from web interface) are 1-based; to account for this, add 1
    #       to unit numbers when reading from .nev file

    # notes:
    # tmask   --> logical vector of same length as rows that is True for rows inside the tslice
    # rows    --> times (in units of s, measured on neural system) that correspond to each row of the task in hdf file
    # kin     --> every 6th row of kinematics within the tslice boundaries
    # neurows --> the rows inside the tslice

    if FAKE_BLACKROCK_TMASK:
       tmask, rows = _get_tmask_blackrock_fake(files['hdf'], tslice)
    else:
        nev_fname = [name for name in files['blackrock'] if '.nev' in name][0]  # only one of them
        tmask, rows = _get_tmask_blackrock(nev_fname, tslice, syskey_fn=lambda x: x[0] in [source, source[1:]]) 
    neurows = rows[tmask]

    neural_features, units, extractor_kwargs = extractor_fn(files, neurows, binlen, units, extractor_kwargs, strobe_rate=strobe_rate)

    return neural_features, units, extractor_kwargs

def get_neural_features(files, binlen, extractor_fn, extractor_kwargs, units=None, tslice=None, source='task', strobe_rate=60):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    
    hdf = tables.openFile(files['hdf'])

    if 'plexon' in files:
        fn = _get_neural_features_plx
    elif 'blackrock' in files:
        fn = _get_neural_features_blackrock
    else:
        raise Exception('Could not find any plexon or blackrock files!')

    neural_features, units, extractor_kwargs = fn(files, binlen, extractor_fn, extractor_kwargs, tslice=tslice, units=units, source=source, strobe_rate=strobe_rate)

    return neural_features, units, extractor_kwargs

################################################################################
## Kinematic data retrieval
################################################################################
def get_plant_pos_vel(files, binlen, tmask, update_rate_hz=60., pos_key='cursor', vel_key=None):
    '''
    Get positions and velocity from 'task' table of HDF file

    Parameters
    ----------

    Returns
    -------
    '''
    if pos_key == 'plant_pos':  # used for ibmi tasks
        vel_key == 'plant_vel'

    hdf = tables.openFile(files['hdf'])    
    kin = hdf.root.task[:][pos_key]    

    inds, = np.nonzero(tmask)
    step_fl = binlen/(1./update_rate_hz)
    if step_fl < 1: # more than one spike bin per kinematic obs
        if vel_key is not None:
            velocity = hdf.root.task[:][vel_key]
        else:
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

        if vel_key is not None:
            velocity = hdf.root.task[inds][vel_key]
        else:
            velocity = np.diff(kin, axis=0) * 1./binlen
            velocity = np.vstack([np.zeros(kin.shape[1]), velocity])
        kin = np.hstack([kin, velocity])

    return kin    


################################################################################
## Main training functions
################################################################################
def create_onedimLFP(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, units, update_rate=0.1, tslice=None, kin_source='task', pos_key='cursor', vel_key=None):
     ## get neural features
    f_extractor = extractor.LFPMTMPowerExtractor(None, **extractor_kwargs)
    import onedim_lfp_decoder as old
    return old.create_decoder(units, ssm, extractor_cls, f_extractor.extractor_kwargs)

def train_KFDecoder(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, units, update_rate=0.1, tslice=None, kin_source='task', pos_key='cursor', vel_key=None):
    '''
    Create a new KFDecoder using maximum-likelihood, from kinematic observations and neural observations

    Parameters
    ---------- 
    files : dict
        Dictionary of files which contain training data. Keys are file tyes, values are file names.
        Kinematic data is assumed to be stored in an 'hdf' file and neural data assumed to be in 'plx' or 'nev' files
    extractor_cls : class
        Class of feature extractor to instantiate
    extractor_kwargs : dict 
        Parameters to specify for feature extractor to instantiate it to specification
    kin_extractor : callable
        Function to extract kinematics from the HDF file.
    ssm : state_space_models.StateSpace instance
        State space model for the Decoder object being created.
    units : np.iterable 
        Spiking units are specified as tuples of (electrode channe, electrode unit)
    update_rate : float, optional
        Time in seconds between decoder updates. default=0.1
    tslice : iterable of length 2, optional
        Start and end times in seconds to specify the portion of the training data to use for ML estimation. By default, the whole dataset will be used
    kin_source : string, optional
        Table from the HDF file to grab kinematic data. Default is the 'task' table.
    pos_key : string, optional
        Column of HDF table to use for position data. Default is 'cursor', recognized options are {'cursor', 'joint_angles', 'plant_pos'}
    vel_key : string
        Column of HDF table to use for velocity data. Default is None; velocity is computed by single-step numerical differencing (or alternate method )

    Returns
    -------
    KFDecoder instance
    '''
    binlen = update_rate

    ## get kinematic data
    tmask, rows = _get_tmask(files, tslice, sys_name=kin_source)
    kin = kin_extractor(files, binlen, tmask, pos_key=pos_key, vel_key=vel_key)

    ## get neural features
    neural_features, units, extractor_kwargs = get_neural_features(files, binlen, extractor_cls.extract_from_file, extractor_kwargs, tslice=tslice, units=units, source=kin_source)

    # Remove 1st kinematic sample and last neural features sample to align the 
    # velocity with the neural features
    kin = kin[1:].T
    neural_features = neural_features[:-1].T

    decoder = train_KFDecoder_abstract(ssm, kin, neural_features, units, update_rate, tslice=tslice)

    decoder.extractor_cls = extractor_cls
    decoder.extractor_kwargs = extractor_kwargs

    return decoder

def train_KFDecoder_abstract(ssm, kin, neural_features, units, update_rate, tslice=None):
    #### Train the actual KF decoder matrices ####
    n_features = neural_features.shape[0]  # number of neural features

    # C should be trained on all of the stochastic state variables, excluding the offset terms
    C = np.zeros([n_features, ssm.n_states])
    C[:, ssm.drives_obs_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(kin[ssm.train_inds, :], neural_features)

    mFR = np.mean(neural_features, axis=1)
    sdFR = np.std(neural_features, axis=1)

    # Set state space model
    A, B, W = ssm.get_ssm_matrices(update_rate=update_rate)

    # instantiate KFdecoder
    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=ssm.is_stochastic)
    decoder = kfdecoder.KFDecoder(kf, units, ssm, mFR=mFR, sdFR=sdFR, binlen=update_rate, tslice=tslice)

    # Compute sufficient stats for C and Q matrices (used for RML CLDA)
    from clda import KFRML
    n_features, n_states = C.shape
    R = np.mat(np.zeros([n_states, n_states]))
    S = np.mat(np.zeros([n_features, n_states]))
    R_small, S_small, T, ESS = KFRML.compute_suff_stats(kin[ssm.train_inds, :], neural_features)

    R[np.ix_(ssm.drives_obs_inds, ssm.drives_obs_inds)] = R_small
    S[:,ssm.drives_obs_inds] = S_small
    
    decoder.filt.R = R
    decoder.filt.S = S
    decoder.filt.T = T
    decoder.filt.ESS = ESS
    decoder.n_features = n_features

    # decoder.extractor_cls = extractor_cls
    # decoder.extractor_kwargs = extractor_kwargs

    return decoder

def train_PPFDecoder(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, units, update_rate=0.1, tslice=None, kin_source='task', pos_key='cursor', vel_key=None):
    '''
    Create a new PPFDecoder using maximum-likelihood, from kinematic observations and neural observations

    Parameters
    ---------- 
    files : dict
        Dictionary of files which contain training data. Keys are file tyes, values are file names.
        Kinematic data is assumed to be stored in an 'hdf' file and neural data assumed to be in 'plx' or 'nev' files
    extractor_cls : class
        Class of feature extractor to instantiate
    extractor_kwargs : dict 
        Parameters to specify for feature extractor to instantiate it to specification
    kin_extractor : callable
        Function to extract kinematics from the HDF file.
    ssm : state_space_models.StateSpace instance
        State space model for the Decoder object being created.
    units : np.iterable 
        Spiking units are specified as tuples of (electrode channe, electrode unit)
    update_rate : float, optional
        Time in seconds between decoder updates. default=0.1
    tslice : iterable of length 2, optional
        Start and end times in seconds to specify the portion of the training data to use for ML estimation. By default, the whole dataset will be used
    kin_source : string, optional
        Table from the HDF file to grab kinematic data. Default is the 'task' table.
    pos_key : string, optional
        Column of HDF table to use for position data. Default is 'cursor', recognized options are {'cursor', 'joint_angles', 'plant_pos'}
    vel_key : string
        Column of HDF table to use for velocity data. Default is None; velocity is computed by single-step numerical differencing (or alternate method )

    Returns
    -------
    PPFDecoder instance
    '''
    binlen = 1./180 #update_rate

    ## get kinematic data
    tmask, rows = _get_tmask(files, tslice, sys_name=kin_source)
    kin = kin_extractor(files, binlen, tmask, pos_key=pos_key, vel_key=vel_key)

    ## get neural features
    neural_features, units, extractor_kwargs = get_neural_features(files, binlen, extractor_cls.extract_from_file, extractor_kwargs, tslice=tslice, units=units, source=kin_source)

    # Remove 1st kinematic sample and last neural features sample to align the 
    # velocity with the neural features
    kin = kin[1:].T
    neural_features = neural_features[:-1].T

    decoder = train_PPFDecoder_abstract(ssm, kin, neural_features, units, update_rate, tslice=tslice)

    decoder.extractor_cls = extractor_cls
    decoder.extractor_kwargs = extractor_kwargs

    return decoder

def train_PPFDecoder_abstract(ssm, kin, neural_features, units, update_rate, tslice=None):
    binlen = 1./180 #update_rate
    # squash any spike counts greater than 1 (doesn't work with PPF model)
    neural_features[neural_features > 1] = 1

    #### Train the  PPF decoder matrices ####
    n_features = neural_features.shape[0]  # number of neural features

    # C should be trained on all of the stochastic state variables, excluding the offset terms
    C = np.zeros([n_features, ssm.n_states])
    C[:, ssm.drives_obs_inds], pvals = ppfdecoder.PointProcessFilter.MLE_obs_model(kin[ssm.train_inds, :], neural_features)

    # Set state space model
    A, B, W = ssm.get_ssm_matrices(update_rate=update_rate)

    # instantiate Decoder
    ppf = ppfdecoder.PointProcessFilter(A, W, C, B=B, dt=update_rate, is_stochastic=ssm.is_stochastic)
    decoder = ppfdecoder.PPFDecoder(ppf, units, ssm, binlen=binlen, tslice=tslice)

    # Compute sufficient stats for C matrix (used for RML CLDA)
    from clda import KFRML
    n_features, n_states = C.shape
    S = np.mat(np.zeros([n_features, n_states]))
    S_small, = decoder.compute_suff_stats(kin[ssm.train_inds, :], neural_features)

    S[:,ssm.drives_obs_inds] = S_small
        
    decoder.filt.S = S
    decoder.n_features = n_features

    return decoder

###################
## Helper functions
###################
def unit_conv(starting_unit, ending_unit):
    ''' Convert between units, e.g. cm to m
    Lookup table for conversion factors between units; this function exists
    only to avoid hard-coded constants in most of the code

    Docstring

    Parameters
    ----------

    Returns
    -------
    '''

    if starting_unit == ending_unit:
        return 1
    elif (starting_unit, ending_unit) == ('cm', 'm'):
        return 0.01
    elif (starting_unit, ending_unit) == ('m', 'cm'):
        return 100

def obj_eq(self, other, attrs=[]):
    '''
    Determine if two objects have mattching array attributes

    Parameters
    ----------
    other : object
        If objects are not the same type, False is returned
    attrs : list, optional
        List of attributes to compare for equality. Only attributes that are common to both objects are used.
        The attributes should be np.array or similar as np.array_equal is used to determine equality

    Returns
    -------
    bool 
        True value returned indicates equality between objects for the specified attributes
    '''
    if isinstance(other, type(self)):
        attrs_eq = filter(lambda y: y in other.__dict__, filter(lambda x: x in self.__dict__, attrs))
        equal = map(lambda attr: np.array_equal(getattr(self, attr), getattr(other, attr)), attrs_eq)
        return np.all(equal)
    else:
        return False
    
def obj_diff(self, other, attrs=[]):
    '''
    Calculate the difference of the two objects w.r.t the specified attributes

    Docstring

    Parameters
    ----------

    Returns
    -------
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

    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    cellname = re.compile(r'(\d{1,3})\s*(\w{1})')
    cells = [ (int(c), ord(u) - 96) for c, u in cellname.findall(cells)]
    return cells

def inflate(A, current_states, full_state_ls, axis=0):
    '''
    'Inflate' a matrix by filling in rows/columns with zeros

    Docstring

    Parameters
    ----------

    Returns
    -------

    '''
    nS = len(full_state_ls)
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


#######################
## Simulation functions
#######################
def _train_PPFDecoder_2D_sim(stochastic_states, neuron_driving_states, units,
    bounding_box, states_to_bound, include_y=True, dt=0.1, v=0.4):
    '''
    Train a simulation PPFDecoder

    Docstring

    Parameters
    ----------

    Returns
    -------
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
    decoder = kfdecoder.KFDecoder(kf, units, _ssm, binlen=dt, n_subbins=1, mFR=mFR, sdFR=sdFR)

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
    decoder.n_features = n_neurons
    return decoder

## added by Sid to as a new version of above function (which seems outdated)
def _train_KFDecoder_2D_sim_2(_ssm, units, dt=0.1):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    n_neurons = units.shape[0]
    binlen = dt

    A, B, W = _ssm.get_ssm_matrices(update_rate=dt)
    drives_neurons = _ssm.drives_obs
    is_stochastic = _ssm.is_stochastic
    nX = _ssm.n_states

    C = np.random.standard_normal([n_neurons, nX])
    C[:, ~drives_neurons] = 0
    Q = 10 * np.identity(n_neurons) 

    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=is_stochastic)

    mFR = 0
    sdFR = 1
    decoder = kfdecoder.KFDecoder(kf, units, _ssm, mFR=mFR, sdFR=sdFR, binlen=binlen)

    decoder.kf.R = np.mat(np.identity(decoder.kf.C.shape[1]))
    decoder.kf.S = decoder.kf.C
    decoder.kf.T = decoder.kf.Q + decoder.kf.S*decoder.kf.S.T
    decoder.kf.ESS = 3000.

    decoder.ssm = _ssm
    decoder.n_features = n_neurons

    decoder.bounder = make_rect_bounder_from_ssm(_ssm)

    return decoder



def rand_KFDecoder(sim_units, state_units='cm'):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
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

    Docstring

    Parameters
    ----------

    Returns
    -------
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

    Docstring

    Parameters
    ----------

    Returns
    -------
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
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
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
