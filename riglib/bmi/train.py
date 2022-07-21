'''
Methods to create various types of Decoder objects from data(files)
'''
import re
import pickle
import sys

import numpy as np
from scipy.io import loadmat
from ..dio import parse

import tables
from . import kfdecoder, ppfdecoder
from . import state_space_models
import os

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
    if sys2 == 'task':
        if sys1 in ['TAS\x00TASK', 'btqassskh', 'btqassskkkh', 'tasktasktask', 'task\x00task\x00task', b'task']:
            return True
        elif sys1[:4] in ['tqas', 'tacs','ttua', 'bttu', 'tttu']:
            return True

    return sys1 in [sys2, sys2[1:], sys2.upper()]


#FAKE_BLACKROCK_TMASK = True
FAKE_BLACKROCK_TMASK = False

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
    elif 'ecube' in files:
        fn = _get_tmask_ecube
        fname = files
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
    if isinstance(plx, str) or isinstance(plx, str):
        plx = plexfile.openFile(plx.encode('utf-8'))

    # Get the list of all the systems registered in the neural data file
    events = plx.events[:].data
    reg = parse.registrations(events)

    if len(list(reg.keys())) > 0:
        # find the key for the specified system data
        syskey = None
        for key, system in list(reg.items()):
            if sys_eq(system[0], sys_name):
                syskey = key
                break

        if syskey is None:
            print((list(reg.items())))
            raise Exception('riglib.bmi.train._get_tmask: Training data source not found in neural data file!')
    elif len(list(reg.keys())) == 0:
        # try to find how many systems' rowbytes were in the HDF file
        rowbyte_data = parse.rowbyte(events)
        if len(list(rowbyte_data.keys())) == 1:
            print("No systems registered, but only one system registered with rowbytes! Using it anyway instead of throwing an error")
            syskey = list(rowbyte_data.keys())[0]
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

    if nev_fname[-4:] != '.hdf':
        nev_hdf_fname = nev_fname + '.hdf'

        if not os.path.isfile(nev_hdf_fname):
            # convert .nev file to hdf file using our own blackrock_parse_files:
            from db.tracker import models
            task_entry = int(nev_fname[-8:-4])
            _, _ = models.parse_blackrock_file(nev_fname, 0, task_entry)
    else:
        nev_hdf_fname = nev_fname

    #import h5py
    #nev_hdf = h5py.File(nev_hdf_fname, 'r')
    nev_hdf = tables.openFile(nev_hdf_fname)

    #path = 'channel/digital00001/digital_set'
    #ts = nev_hdf.get(path).value['TimeStamp']
    #msgs = nev_hdf.get(path).value['Value']

    ts = nev_hdf.root.channel.digital0001.digital_set[:]['TimeStamp']
    msgs = nev_hdf.root.channel.digital0001.digital_set[:]['Value'] + 2**16

    msgtype = np.right_shift(np.bitwise_and(msgs, parse.msgtype_mask), 8).astype(np.uint8)
    # auxdata = np.right_shift(np.bitwise_and(msgs, auxdata_mask), 8).astype(np.uint8)
    auxdata = np.right_shift(np.bitwise_and(msgs, parse.auxdata_mask), 8+3).astype(np.uint8)
    rawdata = np.bitwise_and(msgs, parse.rawdata_mask)

    # data is an N x 4 matrix that will be the argument to parse.registrations()
    data = np.vstack([ts, msgtype, auxdata, rawdata]).T

    # get system registrations
    reg = parse.registrations(data)
    syskey = None

    for key, system in list(reg.items()):
            if sys_eq(system[0], sys_name):
                syskey = key
                break

    if syskey is None:
        raise Exception('No source registration saved in the file!')

    # get the corresponding hdf rows
    rows = parse.rowbyte(data)[syskey][:,0]

    rows = rows / 30000.

    lower, upper = 0 < rows, rows < rows.max() + 1
    if tslice is None:
        l = None;
        u = None;
    else:
        l, u = tslice

    if l is not None:
        lower = l < rows
    if u is not None:
        upper = rows < u
    tmask = np.logical_and(lower, upper)
    return tmask, rows

def _get_tmask_blackrock_fake(hdf_fname, tslice, **kwargs):
    # need to create fake "rows" and "tmask" variables

    print('WARNING: Using _get_tmask_blackrock_fake function!!')

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

def _get_tmask_ecube(files, tslice, **kwargs):
    '''
    Find the "rows" of the ecube file to use for training the decoder.
    For compatibility with plexon and blackrock data, since actually we aren't saving
    any BMI3D rows into the ecube file, only sync'ing the cycle number via a strobe BNC
    output. 

    Args: 
        data_dir: location where the ecube data resides
        tslice : list of length 2
            Specify the start and end time to examine the file, in seconds

    Returns:
        tmask: np.ndarray of shape (N, ) of booleans
            Specifies which BMI3D cycles are within the time bounds
        rows: np.ndarray of shape (N, ) of integers
            The times at which BMI3D cycles were recorded on the ecube
    '''
    from riglib.ecube import load_bmi3d_cycle_times
    rows = load_bmi3d_cycle_times(files)

    # Determine which rows are within the time bounds
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

    hdf = tables.open_file(files['hdf'])

    plx_fname = files['plexon'].encode('utf-8')
    from plexon import plexfile
    try:
        plx = plexfile.openFile(plx_fname)
    except IOError:
        raise Exception("Could not open .plx file: %s" % plx_fname)

    # Use all of the units if none are specified
    if units is None:
        units = np.array(plx.units).astype(np.int32)

    if tslice is None:
        tslice = (1., plx.length-1)

    tmask, rows = _get_tmask_plexon(plx, tslice, sys_name=source)
    neurows = rows[tmask]

    neural_features, units, extractor_kwargs = extractor_fn(files, neurows, binlen, units, extractor_kwargs)

    return neural_features, units, extractor_kwargs

def _get_neural_features_blackrock(files, binlen, extractor_fn, extractor_kwargs, tslice=None, units=None, source='task', strobe_rate=20.):
    if units is None:
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

        #tmask, rows = _get_tmask_blackrock(nev_fname, tslice, syskey_fn=lambda x: x[0] in [source, source[1:]])

        tmask, rows = _get_tmask_blackrock(nev_fname, tslice, sys_name=source)
    neurows = rows[tmask]

    neural_features, units, extractor_kwargs = extractor_fn(files, neurows, binlen, units, extractor_kwargs, strobe_rate=strobe_rate)

    return neural_features, units, extractor_kwargs

def _get_neural_features_tdt(files, binlen, extractor_fn, extractor_kwargs, tslice=None, units=None, source='task', strobe_rate=10.):
    raise NotImplementedError

def _get_neural_features_ecube(files, binlen, extractor_fn, extractor_kwargs, tslice=None, units=None, source='task', strobe_rate=120):
    '''
    Mostly just copied from _get_neural_features_plx()... see their docs
    '''
    from riglib.ecube import parse_file
    info = parse_file(files['ecube'])
    # Use all of the units if none are specified
    if units is None:
        units = np.array(info.units).astype(np.int32)

    if tslice is None:
        tslice = (1., info.length-1)

    tmask, rows = _get_tmask_ecube(files, tslice)
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

    if 'plexon' in files:
        fn = _get_neural_features_plx
    elif 'blackrock' in files:
        fn = _get_neural_features_blackrock
        strobe_rate = 20.
    elif 'tdt' in files:
        fn = _get_neural_features_tdt
    elif 'ecube' in files:
        fn = _get_neural_features_ecube
    else:
        raise Exception('Could not find any recognized neural data files!')

    neural_features, units, extractor_kwargs = fn(files, binlen, extractor_fn, extractor_kwargs, tslice=tslice, units=units, source=source, strobe_rate=strobe_rate)

    return neural_features, units, extractor_kwargs

################################################################################
## Kinematic data retrieval
################################################################################
def null_kin_extractor(files, binlen, tmask, update_rate_hz=60., pos_key='cursor', vel_key=None):
    hdf = tables.openFile(files['hdf'])
    kin = np.squeeze(hdf.root.task[:][pos_key])

    inds, = np.nonzero(tmask)
    step_fl = binlen/(1./update_rate_hz)
    if step_fl < 1: # more than one spike bin per kinematic obs
        kin = np.hstack([kin, velocity])

        n_repeats = int((1./update_rate_hz)/binlen)
        inds = np.sort(np.hstack([inds]*n_repeats))
        kin = kin[inds]
    else:
        step = int(binlen/(1./update_rate_hz))
        inds = inds[::step]
        kin = kin[inds]

    print(("kin.shape", kin.shape))
    return kin


def get_plant_pos_vel(files, binlen, tmask, update_rate_hz=60., pos_key='cursor', vel_key=None):
    '''
    Get positions and velocity from 'task' table of HDF file

    Parameters
    ----------

    Returns
    -------
    '''
    if pos_key == 'plant_pos':  # used for ibmi tasks
        vel_key = 'plant_vel'

    hdf = tables.open_file(files['hdf'])
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
        try:
            kin = kin[inds]
            if vel_key is not None:
                velocity = hdf.root.task[inds][vel_key]
            else:
                velocity = np.diff(kin, axis=0) * 1./binlen
                velocity = np.vstack([np.zeros(kin.shape[1]), velocity])

        except:
            kin2 = np.zeros((len(inds), kin.shape[1]))
            vel2 = np.zeros((len(inds), kin.shape[1]))

            ix = np.nonzero(inds < len(kin))[0]
            kin2[ix, :] = kin[inds[ix], :]
            kin = kin2.copy()

            if vel_key is not None:
                vel2[ix, :] = hdf.root.task[inds[ix]][vel_key]
            else:
                vel2 = np.diff(kin, axis=0) * 1./binlen
                vel2 = np.vstack([np.zeros(kin.shape[1]), vel2])

            velocity = vel2.copy()

        kin = np.hstack([kin, velocity])

    return kin


################################################################################
## Main training functions
################################################################################
def create_onedimLFP(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, units, update_rate=0.1, tslice=None, kin_source='task',
    pos_key='cursor', vel_key=None, zscore=False):
    ## get neural features
    from . import extractor
    f_extractor = extractor.LFPMTMPowerExtractor(None, **extractor_kwargs)
    from . import onedim_lfp_decoder as old
    return old.create_decoder(units, ssm, extractor_cls, f_extractor.extractor_kwargs)

def test_ratBMIdecoder(te_id=None, update_rate=0.1, tslice=None, kin_source='task', pos_key='cursor', vel_key=None, **kwargs):
    from db import dbfunctions as dbfn
    te = dbfn.TaskEntry(te_id)
    files = dict(hdf=te.hdf_filename, plexon=te.plx_filename)

    entry = te.id
    from . import extractor
    extractor_cls = extractor.BinnedSpikeCountsExtractor

    neural_features, units, extractor_kwargs = get_neural_features(files, 0.1, extractor_cls.extract_from_file, dict(), tslice=None)
    extractor_kwargs['units'] = units

    from . import rat_bmi_decoder
    nsteps = kwargs.pop('nsteps', 10)
    prob_t1 = kwargs.pop('prob_t1', 0.985)
    prob_t2 = kwargs.pop('prob_t2', 0.015)
    timeout = kwargs.pop('timeout', 30.)
    timeout_pause = kwargs.pop('timeout_pause', 3.)
    freq_lim = kwargs.pop('freq_lim', (1000., 20000.))
    e1_inds = kwargs.pop('e1_inds', None)
    e2_inds = kwargs.pop('e2_inds', None)

    e1_inds, e2_inds, FR_to_freq_fn, units, t1, t2, mid = rat_bmi_decoder.calc_decoder_from_baseline_file(neural_features, units, nsteps, prob_t1, prob_t2, timeout,
        timeout_pause, freq_lim, e1_inds, e2_inds)

    task_params = dict(nsteps=nsteps, prob_t1=prob_t1, prob_t2=prob_t2, timeout_pause=timeout_pause, timeout=timeout, freq_lim=freq_lim,
        e1_inds=e1_inds, e2_inds=e2_inds, te_name=te.name, FR_to_freq_fn=FR_to_freq_fn, units=units, te_id=te_id, t1=t1, t2=t2, mid=mid,
        extractor_kwargs=extractor_kwargs)

    return task_params

def test_IsmoreSleepDecoder(te_id, e1_units, e2_units, nsteps=1, prob_t1 = 0.985, prob_t2 = 0.015, timeout = 15.,
    timeout_pause=0., freq_lim = [-1, 1], targets_matrix=None, session_length=0, saturate_perc=90,
    skip_sim=False):

    from db import dbfunctions as dbfn
    te = dbfn.TaskEntry(te_id)
    files = dict(hdf=te.hdf_filename, blackrock=te.blackrock_filenames)
    entry = te.id
    from . import extractor
    extractor_cls = extractor.BinnedSpikeCountsExtractor

    units = np.vstack((e1_units, e2_units))
    argsort = np.argsort(units[:, 0])
    units = units[argsort, :]

    unit_ids = np.hstack((['e1']*len(e1_units) + ['e2']*len(e2_units)))
    sorted_unit_ids = unit_ids[argsort]

    e1_inds = np.nonzero(sorted_unit_ids=='e1')[0]
    e2_inds = np.nonzero(sorted_unit_ids=='e2')[0]

    neural_features, units, extractor_kwargs = get_neural_features(files, 0.1, extractor_cls.extract_from_file,
        dict(), tslice=None, units=units)

    neural_features_unbinned, units, extractor_kwargs = get_neural_features(files, 0.05, extractor_cls.extract_from_file,
        dict(), tslice=None, units=units)

    import riglib.bmi.rat_bmi_decoder

    kwargs = dict(targets_matrix=targets_matrix, session_length=session_length,
        saturate_perc=saturate_perc, skip_sim=skip_sim)

    decoder, nrewards = riglib.bmi.rat_bmi_decoder.calc_decoder_from_baseline_file(neural_features,
        neural_features_unbinned, units, nsteps, prob_t1, prob_t2, timeout, timeout_pause, freq_lim,
        e1_inds, e2_inds, sim_fcn='ismore', **kwargs)

    decoder.extractor_cls = extractor_cls
    decoder.extractor_kwargs = extractor_kwargs
    pickle.dump(decoder, open('/storage/decoders/sleep_from_te'+str(te_id)+'.pkl', 'wb'))
    from db.tracker import dbq
    dbq.save_bmi('sleep_from_te'+str(te_id), te_id, '/storage/decoders/sleep_from_te'+str(te_id)+'.pkl')
    return decoder, nrewards

def create_ratBMIdecoder(task_params):
    from . import extractor
    task_params['extractor_cls'] = extractor.BinnedSpikeCountsExtractor
    from . import rat_bmi_decoder
    from . import state_space_models
    rat_decoder= rat_bmi_decoder.create_decoder(state_space_models.StateSpaceEndptPos1D(), task_params)
    rat_decoder.extractor_kwargs = task_params['extractor_kwargs']
    import tempfile
    import pickle
    from db.tracker import dbq

    rat_decoder.te_id = task_params['te_id']
    tf = tempfile.NamedTemporaryFile('wb')
    pickle.dump(rat_decoder, tf, 2)
    tf.flush()

    name = task_params['te_name'] + '_rat_bmi_decoder'
    dbq.save_bmi(name, int(task_params['te_id']), tf.name)

def create_lindecoder(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, units=None, update_rate=0.1, tslice=None, kin_source='task',
    pos_key='cursor', vel_key=None, zscore=False):
    from . import lindecoder, state_space_models
    
    # Hack job incoming:
    if 'mouse' in files:
        neural_data = [[0., 0.], [1650., 1080.]]
        units = [(1, 0), (2, 0)]
        unit_to_state = None
        decoder_to_plant = 20
        smoothing_window = 1
        vel_control = False
    return lindecoder.create_lindecoder(ssm, units, neural_data, unit_to_state, decoder_to_plant, smoothing_window, vel_control, update_rate)

def add_fa_dict_to_decoder(decoder_training_te, dec_ix, fa_te):
    #First make sure we're training from the correct task entry: spike counts n_units == BMI units
    from db import dbfunctions as dbfn
    te = dbfn.TaskEntry(fa_te)
    hdf = te.hdf
    sc_n_units = hdf.root.task[0]['spike_counts'].shape[0]


    from db.tracker import models
    te_arr = models.Decoder.objects.filter(entry=decoder_training_te)
    search_flag = 1
    for te in te_arr:
        ix = te.path.find('_')
        if search_flag:
            if int(te.path[ix+1:ix+3]) == dec_ix:
                decoder_old = te
                search_flag = 0

    if search_flag:
        raise Exception('No decoder from ', str(decoder_training_te), ' and matching index: ', str(dec_ix))

    from tasks.factor_analysis_tasks import FactorBMIBase
    FA_dict = FactorBMIBase.generate_FA_matrices(fa_te)

    import pickle
    dec = pickle.load(open(decoder_old.filename))
    dec.trained_fa_dict = FA_dict
    dec_n_units = dec.n_units

    if dec_n_units != sc_n_units:
        raise Exception('Cant use TE for BMI training and FA training -- n_units mismatch')

    from db import trainbmi
    trainbmi.save_new_decoder_from_existing(dec, decoder_old, suffix='_w_fa_dict_from_'+str(fa_te))

def train_FADecoder_from_KF(FA_nfactors, FA_te_id, decoder, use_scaled=True, use_main=True):

    from tasks.factor_analysis_tasks import FactorBMIBase
    FA_dict = FactorBMIBase.generate_FA_matrices(FA_nfactors, FA_te_id)

    # #Now, retrain:
    binlen = decoder.binlen

    from db import dbfunctions as dbfn
    te_id = dbfn.TaskEntry(decoder.te_id)
    files = dict(plexon=te_id.plx_filename, hdf = te_id.hdf_filename)
    extractor_cls = decoder.extractor_cls
    extractor_kwargs = decoder.extractor_kwargs
    kin_extractor = get_plant_pos_vel
    ssm = decoder.ssm
    update_rate = decoder.binlen
    units = decoder.units
    tslice = (0., te_id.length)

    ## get kinematic data
    kin_source = 'task'
    tmask, rows = _get_tmask(files, tslice, sys_name=kin_source)
    kin = kin_extractor(files, binlen, tmask, pos_key='cursor', vel_key=None)

    ## get neural features
    neural_features, units, extractor_kwargs = get_neural_features(files, binlen, extractor_cls.extract_from_file, extractor_kwargs, tslice=tslice, units=units, source=kin_source)

    #Get shared input:
    T = neural_features.shape[0]
    demean = neural_features.T - np.tile(FA_dict['fa_mu'], [1, T])

    if use_main:
        main_shar = (FA_dict['fa_main_shared'] * demean)
        main_priv = (demean - main_shar)
        FA = FA_dict['FA_model']

    else:
        shar = (FA_dict['fa_sharL']* demean)
        shar_sc = np.multiply(shar, np.tile(FA_dict['fa_shar_var_sc'], [1, T])) + np.tile(FA_dict['fa_mu'], [1, T])
        shar_unsc = shar + np.tile(FA_dict['fa_mu'], [1, T])
        if use_scaled:
            neural_features = shar_sc[:,:-1]
        else:
            neural_features = shar_unsc[:,:-1]

    # Remove 1st kinematic sample and last neural features sample to align the
    # velocity with the neural features
    kin = kin[1:].T

    decoder2 = train_KFDecoder_abstract(ssm, kin, neural_features, units, update_rate, tslice=tslice)
    decoder2.extractor_cls = extractor_cls
    decoder2.extractor_kwargs = extractor_kwargs
    decoder2.te_id = decoder.te_id
    decoder2.trained_fa_dict = FA_dict

    import datetime
    now = datetime.datetime.now()
    tp = now.isoformat()
    import pickle
    fname = os.path.expandvars('$FA_GROM_DATA/decoder_')+tp+'.pkl'
    f = open(fname, 'w')
    pickle.dump(decoder2, f)
    f.close()
    return decoder2, fname

def conv_KF_to_splitFA_dec(decoder_training_te, dec_ix, fa_te, search_suffix = 'w_fa_dict_from_', use_shar_z=False, tslice=None):

    from db import dbfunctions as dbfn
    te = dbfn.TaskEntry(fa_te)
    hdf = te.hdf
    sc_n_units = hdf.root.task[0]['spike_counts'].shape[0]

    from db.tracker import models
    te_arr = models.Decoder.objects.filter(entry=decoder_training_te)
    search_flag = 1
    for te in te_arr:
        ix = te.path.find('_')
        if search_flag:
            if int(te.path[ix+1:ix+3]) == dec_ix:
                decoder = pickle.load(open(te.filename))
                if hasattr(decoder, 'trained_fa_dict'):
                    ix = te.path.find('w_fa_dict_from_')
                    if ix > 1:
                        fa_te_train = te.path[ix+len(search_suffix):ix+len(search_suffix)+4]
                        if int(fa_te_train) == fa_te:
                            decoder_old = te
                            #search_flag = 0

    # if search_flag:
    #     raise Exception('No decoder from ', str(decoder_training_te), ' and matching index: ', str(dec_ix), ' with FA training from: ',str(fa_te))
    # else:
    print(('Using old decoder: ', decoder_old.path))

    decoder = pickle.load(open(decoder_old.filename))
    if hasattr(decoder, 'trained_fa_dict'):
        FA_dict = decoder.trained_fa_dict
    else:
        raise Exception('Make an FA dict decoder first, then re-train that')

    from db import dbfunctions as dbfn
    te_id = dbfn.TaskEntry(fa_te)

    files = dict(plexon=te_id.plx_filename, hdf = te_id.hdf_filename)
    extractor_cls = decoder.extractor_cls
    extractor_kwargs = decoder.extractor_kwargs
    extractor_kwargs['discard_zero_units'] = False
    kin_extractor = get_plant_pos_vel
    ssm = decoder.ssm
    update_rate = binlen = decoder.binlen
    units = decoder.units
    if tslice is None:
        tslice = (0., te_id.length)

    ## get kinematic data
    kin_source = 'task'
    tmask, rows =_get_tmask(files, tslice, sys_name=kin_source)
    kin = kin_extractor(files, binlen, tmask, pos_key='cursor', vel_key=None)

    ## get neural features
    neural_features, units, extractor_kwargs = get_neural_features(files, binlen, extractor_cls.extract_from_file, extractor_kwargs, tslice=tslice, units=units, source=kin_source)

    #Get main shared input:
    T = neural_features.shape[0]
    demean = neural_features.T - np.tile(FA_dict['fa_mu'], [1, T])

    #Neural features in time x spikes:
    FA = FA_dict['FA_model']
    z = FA.transform(demean.T)
    z = z.T
    z = z[:FA_dict['fa_main_shar_n_dim'], :]

    #z = FA_dict['u_svd'].T*FA_dict['uut_psi_inv']*demean

    shar_z = FA_dict['fa_main_shared'] * demean
    priv = demean - shar_z

    #Time by features:
    if use_shar_z:
        neural_features2 = np.vstack((z, priv))
        suffx = '_split_shar_z'
    else:
        neural_features2 = np.vstack((z, priv))
        suffx = '_split_z'
    decoder_split = train_KFDecoder_abstract(ssm, kin.T, neural_features2, units, update_rate, tslice=tslice)
    decoder_split.n_features = len(units)
    decoder_split.trained_fa_dict = FA_dict

    decoder_split.extractor_cls = extractor_cls
    decoder_split.extractor_kwargs = extractor_kwargs

    from db import trainbmi
    trainbmi.save_new_decoder_from_existing(decoder_split, decoder_old, suffix=suffx)

def train_KFDecoder(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, units, update_rate=0.1, tslice=None,
    kin_source='task', pos_key='cursor', vel_key=None, zscore=False, filter_kin=False, simple_lin_reg=False,
    use_data_kwargs=None, update_rate_hz=60, **kwargs):
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
    zscore : Bool
        Determines whether to zscore neural_data or not
    kwargs:
        mFR: mean firing rate to use to zscore units
        sdFR: standard dev. to use to zscore units

    Returns
    -------
    KFDecoder instance
    '''
    import sys
    print(files)
    # sys.stdout.write(files)
    # sys.stdout.write(extractor_cls)
    # sys.stdout.write(extractor_kwargs.keys())
    # sys.stdout.write(units)
    binlen = update_rate

    ## get kinematic data
    tmask, rows = _get_tmask(files, tslice, sys_name=kin_source)
    kin = kin_extractor(files, binlen, tmask, pos_key=pos_key, vel_key=vel_key, update_rate_hz=update_rate_hz)

    ## get neural features
    if 'blackrock' in list(files.keys()):
        strobe_rate = 20.
    elif 'ecube' in list(files.keys()):
        strobe_rate = update_rate_hz # they are always the same
    else:
        strobe_rate = 60.
    
    neural_features, units, extractor_kwargs = get_neural_features(files, binlen, extractor_cls.extract_from_file,
        extractor_kwargs, tslice=tslice, units=units, source=kin_source, strobe_rate=strobe_rate)

    # Remove 1st kinematic sample and last neural features sample to align the
    # velocity with the neural features
    kin = kin[1:].T
    neural_features = neural_features[:-1].T

    if filter_kin:
        n_channels = len(kin)
        filts = get_filterbank(fs=1./update_rate, n_channels=n_channels)
        kin_filt = np.zeros_like(kin)
        for chan in range(n_channels):
            for filt in filts[chan]:
                kin_filt[chan, :] = filt(kin[chan, :])
    else:
        kin_filt = kin.copy()

    if simple_lin_reg:
        from sklearn.linear_model import Ridge
        decoder = Ridge(1000.0, fit_intercept=True, normalize=False)

        if use_data_kwargs is not None:

            # HDF rows to use in training
            X = []
            Y = []

            for pair in use_data_kwargs['pairs']:
                X.append(neural_features[:, pair[0]])
                Y.append(kin_filt[:, pair[1]])

            # Convert these hdf rows to
        decoder.fit(np.vstack((X)), np.vstack((Y)))

    else:
        decoder = train_KFDecoder_abstract(ssm, kin_filt, neural_features, units, update_rate, tslice=tslice, zscore=zscore, **kwargs)
        decoder.extractor_cls = extractor_cls
        decoder.extractor_kwargs = extractor_kwargs

    return decoder #, neural_features, kin_filt

def get_filterbank(n_channels=14, fs=1000.):
    # from ismore.filter import Filter
    from scipy.signal import butter, filtfilt
    band  = [.001, 1]  # Hz
    nyq   = 0.5 * fs
    low   = band[0] / nyq
    high  = band[1] / nyq
    high = np.min([high, 0.99])
    bpf_coeffs = butter(4, [low, high], btype='band')

    channel_filterbank = [None]*n_channels
    for k in range(n_channels):
        # filts = [Filter(bpf_coeffs[0], bpf_coeffs[1])]
        filts = [lambda x: filtfilt(bpf_coeffs[0], bpf_coeffs[1], x)]
        channel_filterbank[k] = filts
    return channel_filterbank


def train_KFDecoderDrift(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, units, update_rate=0.1, tslice=None,
    kin_source='task', pos_key='cursor', vel_key=None, zscore=False, update_rae_hz=60, **kwargs):
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
    zscore : Bool
        Determines whether to zscore neural_data or not
    kwargs:
        mFR: mean firing rate to use to zscore units
        sdFR: standard dev. to use to zscore units

    Returns
    -------
    KFDecoder instance
    '''
    import sys
    print(files)
    # sys.stdout.write(files)
    # sys.stdout.write(extractor_cls)
    # sys.stdout.write(extractor_kwargs.keys())
    # sys.stdout.write(units)
    binlen = update_rate

    ## get kinematic data
    tmask, rows = _get_tmask(files, tslice, sys_name=kin_source)
    kin = kin_extractor(files, binlen, tmask, pos_key=pos_key, vel_key=vel_key, update_rate_hz=update_rate_hz)

    ## get neural features
    neural_features, units, extractor_kwargs = get_neural_features(files, binlen, extractor_cls.extract_from_file,
        extractor_kwargs, tslice=tslice, units=units, source=kin_source)

    # Remove 1st kinematic sample and last neural features sample to align the
    # velocity with the neural features
    kin = kin[1:].T
    neural_features = neural_features[:-1].T

    kwargs['driftKF'] = True
    decoder = train_KFDecoder_abstract(ssm, kin, neural_features, units, update_rate,
        tslice=tslice, zscore=zscore, **kwargs)

    decoder.extractor_cls = extractor_cls
    decoder.extractor_kwargs = extractor_kwargs

    return decoder

def train_KFDecoder_abstract(ssm, kin, neural_features, units, update_rate, tslice=None, regularizer=0.,
    zscore=False, **kwargs):
    print(kwargs)
    print('end of kwargs')

    #### Train the actual KF decoder matrices ####
    if type(zscore) is bool:
        pass
    else:
        if zscore == 'on':
            zscore = True
        elif zscore == 'off':
            zscore = False
        else:
            raise Exception

    print(('zscore value: ', zscore, type(zscore)))

    if zscore:
        if 'mFR' in kwargs and 'sdFR' in kwargs:
            print('using kwargs mFR, sdFR to zscore')
            mFR = kwargs['mFR']
            sdFR = kwargs['sdFR']
        else:
            print('computing own mFR, sdFR to zscore')
            mFR = np.mean(neural_features, axis=1)
            sdFR = np.std(neural_features, axis=1)
            if hasattr(kwargs, 'zscore_set_std_to_one'):
                sdFR = np.ones_like(mFR)
        neural_features = (neural_features - mFR[:, np.newaxis])*(1./sdFR[:, np.newaxis])

    else:
        mFR = np.squeeze(np.mean(neural_features, axis=1))
        sdFR = np.squeeze(np.std(neural_features, axis=1))

    if 'noise_rej' in kwargs:
        if kwargs['noise_rej']:
            sum_pop = np.sum(neural_features, axis = 0)
            bins_noisy = np.nonzero(sum_pop > kwargs['noise_rej_cutoff'])[0]
            print(('replacing %d noisy bins of total %d bins w/ mFR for decoder training!' % (len(bins_noisy), len(sum_pop))))
            neural_features[:, bins_noisy] = mFR[:, np.newaxis]
    else:
        kwargs['noise_rej'] = False
        kwargs['noise_rej_cutoff'] = -1.

    n_features = len(mFR)

    # C should be trained on all of the stochastic state variables, excluding the offset terms
    C = np.zeros((n_features, ssm.n_states))
    C[:, ssm.drives_obs_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(kin[ssm.train_inds, :], neural_features, regularizer=regularizer)


    # Set state space model
    A, B, W = ssm.get_ssm_matrices(update_rate=update_rate)

    # instantiate KFdecoder
    driftKF = kwargs.pop('driftKF', False)
    if driftKF:
        print(('Training Drift Decoder. Noise Rejection? ', kwargs['noise_rej']))
        kf = kfdecoder.KalmanFilterDriftCorrection(A, W, C, Q, is_stochastic=ssm.is_stochastic)
    else:
        kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=ssm.is_stochastic)

    decoder = kfdecoder.KFDecoder(kf, units, ssm, binlen=update_rate, tslice=tslice)

    if zscore:
        decoder.init_zscore(mFR, sdFR)
        print('zscore init')
    else:
        print('no init_zscore')


    # Compute sufficient stats for C and Q matrices (used for RML CLDA)
    from .clda import KFRML
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

    decoder.filt.noise_rej = kwargs['noise_rej']
    decoder.filt.noise_rej_cutoff = kwargs['noise_rej_cutoff']
    decoder.filt.noise_rej_mFR = mFR
    # decoder.extractor_cls = extractor_cls
    # decoder.extractor_kwargs = extractor_kwargs

    return decoder

def train_PPFDecoder(files, extractor_cls, extractor_kwargs, kin_extractor, ssm, units, update_rate=0.1, tslice=None, kin_source='task',
    pos_key='cursor', vel_key=None, zscore=False):
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
    from .clda import KFRML
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
    '''
    Convert between units, e.g. cm to m
    Lookup table for conversion factors between units; this function exists
    only to avoid hard-coded constants in most of the code

    Parameters
    ----------
    starting_unit : string
        Name of current unit for the quantity, e.g., 'cm'
    ending_unit : string
        Name of desired unit for the quantity, e.g., 'm'

    Returns
    -------
    float
        Multiplicative scale factor to convert a scalar in the 'starting_unit' to the 'ending_unit'
    '''

    if starting_unit == ending_unit:
        return 1
    elif (starting_unit, ending_unit) == ('cm', 'm'):
        return 0.01
    elif (starting_unit, ending_unit) == ('m', 'cm'):
        return 100
    else:
        raise ValueError("Unrecognized starting/ending unit")

def lookup_cells(cells):
    '''
    Convert string names of units to 'machine' format.
    Take a list of neural units specified as a list of strings and convert
    to the 2D array format used to specify neural units to train decoders

    Parameters
    ----------
    cells : string
        String of cell names to parse, e.g., '1a, 2b'

    Returns
    -------
    list of 2-tuples
        Each element of the list is a tuple of (channel, unit), e.g., [(1, 1), (2, 2)]
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
    try:
        nS = len(full_state_ls)
    except:
        nS = full_state_ls.n_states

    if axis == 0:
        A_new = np.zeros([nS, A.shape[1]])
    elif axis == 1:
        A_new = np.zeros([A.shape[0], nS])

    try:
        new_inds = [full_state_ls.index(x) for x in current_states]
    except:
        new_inds = [full_state_ls.state_names.index(x) for x in current_states]
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

def rand_KFDecoder(ssm, units, dt=0.1):
    '''
    Make a KFDecoder with the observation model initialized randomly

    Parameters
    ----------
    ssm : state_space_models.StateSpace instance
        State-space model for the KFDecoder. Should specify the A and W matrices
    units : np.array of shape (N, 2)
        Unit labels to assign to each row of the C matrix

    Returns
    -------
    KFDecoder instance
    '''
    n_neurons = units.shape[0]
    binlen = dt

    A, B, W = ssm.get_ssm_matrices(update_rate=dt)
    drives_neurons = ssm.drives_obs
    is_stochastic = ssm.is_stochastic
    nX = ssm.n_states

    C = np.random.standard_normal([n_neurons, nX])
    C[:, ~drives_neurons] = 0
    Q = 10 * np.identity(n_neurons)

    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=is_stochastic)

    mFR = 0
    sdFR = 1
    decoder = kfdecoder.KFDecoder(kf, units, ssm, mFR=mFR, sdFR=sdFR, binlen=binlen)

    decoder.kf.R = np.mat(np.identity(decoder.kf.C.shape[1]))
    decoder.kf.S = decoder.kf.C
    decoder.kf.T = decoder.kf.Q + decoder.kf.S*decoder.kf.S.T
    decoder.kf.ESS = 3000.

    decoder.ssm = ssm
    decoder.n_features = n_neurons

    # decoder.bounder = make_rect_bounder_from_ssm(ssm)

    return decoder

_train_KFDecoder_2D_sim_2 = rand_KFDecoder

def make_fixed_kf_decoder(units, ssm, C, dt=0.1):
    n_neurons = units.shape[0]
    assert n_neurons == C.shape[0], "C matrix must have same first dimension as number of neurons"
    binlen = dt

    A, B, W = ssm.get_ssm_matrices(update_rate=dt)
    drives_neurons = ssm.drives_obs
    is_stochastic = ssm.is_stochastic
    nX = ssm.n_states
    assert nX == C.shape[1], "C matrix must have same second dimension as number of states"

    Q = 10 * np.identity(n_neurons)

    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=is_stochastic)

    mFR = 0
    sdFR = 1
    decoder = kfdecoder.KFDecoder(kf, units, ssm, mFR=mFR, sdFR=sdFR, binlen=binlen)

    decoder.kf.R = np.mat(np.identity(decoder.kf.C.shape[1]))
    decoder.kf.S = decoder.kf.C
    decoder.kf.T = decoder.kf.Q + decoder.kf.S*decoder.kf.S.T
    decoder.kf.ESS = 3000.

    decoder.ssm = ssm
    decoder.n_features = n_neurons

    # decoder.bounder = make_rect_bounder_from_ssm(ssm)

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

    ssm = state_space_models.StateSpaceEndptVel2D()
    A, _, W = ssm.get_ssm_matrices(update_rate=dt)

    # rescale beta for units
    beta[:,3:6] *= units_mult

    # Control input matrix for SSM for control inputs
    I = np.mat(np.eye(3))
    B = np.vstack([0*I, dt*1000 * I, np.zeros([1,3])])

    # instantiate Decoder
    ppf = ppfdecoder.PointProcessFilter(A, W, beta, dt=dt, is_stochastic=ssm.is_stochastic, B=B)
    dec = ppfdecoder.PPFDecoder(ppf, units, ssm, binlen=dt)

    n_stoch_states = len(np.nonzero(ssm.drives_obs)[0])
    n_units = len(units)
    dec.H = np.dstack([np.eye(3)*100] * n_units).transpose(2, 0, 1)
    dec.M = np.mat(np.ones([n_units, n_stoch_states])) * np.exp(-1.6)
    dec.S = np.mat(np.ones([n_units, n_stoch_states])) * np.exp(-1.6)

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
    #beta_full = inflate(beta, states_explaining_neural_activity_2D_vel_decoding, states_3D_endpt, axis=1)
    #states = states_3D_endpt#['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
    #states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
    states = state_space_models.StateSpaceEndptVel2D()
    neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
    beta_full = inflate(beta, neuron_driving_states, states, axis=1)

    stochastic_states = ['hand_vx', 'hand_vz']
    try:
        is_stochastic = [x in stochastic_states for x in states]
    except:
        is_stochastic = [x in stochastic_states for x in states.state_names]

    unit_names = [str(x[0]) for x in data['decoder']['predSig'][0,0][0]]
    units = [(int(x[3:6]), ord(x[-1]) - (ord('a') - 1)) for x in unit_names]
    units = np.vstack(units)

    ppf = ppfdecoder.PointProcessFilter(A, W, beta_full, dt, is_stochastic=is_stochastic)
    ppf.spike_rate_dt = spike_rate_dt

    try:
        drives_neurons = np.array([x in neuron_driving_states for x in states])
    except:
        drives_neurons = np.array([x in neuron_driving_states for x in states.state_names])
    dec = ppfdecoder.PPFDecoder(ppf, units, states, binlen=dt)

    if state_units == 'cm':
        dec.filt.W[3:6, 3:6] *= unit_conv('m', state_units)**2
    return dec
