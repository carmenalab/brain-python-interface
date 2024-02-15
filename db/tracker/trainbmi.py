'''
Functions to call appropriate constructor functions based on UI data and to link decoder objects in the database
'''
import os
import re
import tempfile
import xmlrpc.client
import pickle
import numpy as np

from celery import chain
from .celery import app

from riglib.bmi import extractor

@app.task
def cache_plx(plxfile):
    """
    Create cache for plexon file
    """
    from plexon import plexfile
    plexfile.openFile(plxfile.encode('utf-8'))

@app.task
def make_bmi(name, clsname, extractorname, entry, cells, channels, binlen, tslice, ssm, pos_key, kin_extractor, zscore):
    """
    Create a new Decoder object from training data and save a record to the database

    Parameters
    ----------
    name : string
        Name assigned to decoder object in the database
    clsname : string
        BMI algorithm name (passed to bmilist lookup table 'bmis')
    extractorname : string
        feature extractor algorithm name (passed to bmilist lookup table 'extractors')
    entry : models.TaskEntry
        Django record of training task
    cells : string
        Single string containing all the units to be in decoder, matching
        format in global regex 'cellname' (used only for spike extractors)
    channels : string
        Single string containing all the channels to be in decoder; must be a
        comma separated list of values with spaces (e.g., "1, 2, 3")
        (used only for, e.g., LFP extractors)
    binlen : float
        Time of spike history to consider
    tslice : slice
        Task time to use when training the decoder
    ssm : string
        TODO
    pos_key : string
        TODO
    """
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
    from . import models
    from .json_param import Parameters
    from .tasktrack import Track
    from config import bmiconfig as namelist

    cellname = re.compile(r'(\d{1,3})\s*(\w{1})')

    print("make bmi")

    extractor_cls = namelist.extractors[extractorname]
    print('Training with extractor class:', extractor_cls)

    if 'spike' in extractor_cls.feature_type:  # e.g., 'spike_counts'
        # look at "cells" argument (ignore "channels")

        cells = [ (int(c), ord(u) - 96) for c, u in cellname.findall(cells)]
        if cells == []:
            units = None  # use all units by default
            # Note: inside training functions (e.g., _train_KFDecoder_manual_control,
            #   _train_KFDecoder_visual_feedback, etc.), remember to check if units
            #   variable is None, and if so, set the units from the plx file:
            #       if units == None:
            #           units = np.array(plx.units).astype(np.int32)"
        else:
            unique_cells = []
            for c in cells:
                if c not in unique_cells:
                    unique_cells.append(c)

            units = np.array(unique_cells).astype(np.int32)
    elif ('lfp' in extractor_cls.feature_type) or ('ai_' in extractor_cls.feature_type):  # e.g., 'lfp_power'
        # look at "channels" argument (ignore "cells")
        channels = np.array(channels.split(', ')).astype(np.int32)  # convert str to list of numbers
        if len(channels) == 0:
            channels = [1, 2, 3, 4]  # use these channels by default
        else:
            channels = np.unique(channels)
            #  units = np.hstack([channels.reshape(-1, 1), np.zeros(channels.reshape(-1, 1).shape, dtype=np.int32)])
            units = np.hstack([channels.reshape(-1, 1), np.ones(channels.reshape(-1, 1).shape, dtype=np.int32)])
    elif ('obs' in extractor_cls.feature_type):
        # ignore units and channels
        units = None
    else:
        raise Exception('Unknown extractor class!')

    # task_update_rate = 60 # NOTE may not be true for all tasks?!
    entry_data = models.TaskEntry.objects.get(id=entry).to_json()
    if hasattr(entry_data, 'params') and hasattr(entry_data['params'], 'fps'):
        task_update_rate = entry_data['params']['fps']
    else:
        task_update_rate = 60.

    extractor_kwargs = dict()
    if extractor_cls == extractor.BinnedSpikeCountsExtractor:
        extractor_kwargs['units'] = units
        extractor_kwargs['n_subbins'] = max(1, int((1./task_update_rate)/binlen))
    elif extractor_cls == extractor.LFPButterBPFPowerExtractor:
        extractor_kwargs['channels'] = channels
    elif extractor_cls == extractor.LFPMTMPowerExtractor:
        extractor_kwargs['channels'] = channels
    elif extractor_cls == extractor.AIMTMPowerExtractor:
        extractor_kwargs['channels'] = channels
    elif extractor_cls == extractor.DirectObsExtractor:
        pass
    else:
        raise Exception("Unknown extractor_cls: %s" % extractor_cls)

    # list of DataFile objects
    datafiles = models.DataFile.objects.filter(entry_id=entry)

    # key: a string representing a system name (e.g., 'plexon', 'blackrock', 'task', 'hdf')
    # value: a single filename, or a list of filenames if there are more than one for that system
    files = dict()
    system_names = set(d.system.name for d in datafiles)
    for system_name in system_names:
        filenames = [d.get_path() for d in datafiles if d.system.name == system_name]
        if system_name in ['blackrock', 'blackrock2']:
            files[system_name] = filenames  # list of (one or more) files
        else:
            assert(len(filenames) == 1)
            files[system_name] = filenames[0]  # just one file

    training_method = namelist.bmi_algorithms[clsname]
    ssm = namelist.bmi_state_space_models[ssm]
    kin_extractor_fn = namelist.kin_extractors[kin_extractor]
    decoder = training_method(files, extractor_cls, extractor_kwargs, kin_extractor_fn, ssm, units, update_rate=binlen, tslice=tslice, pos_key=pos_key,
        zscore=zscore, update_rate_hz=task_update_rate)
    decoder.te_id = entry

    tf = tempfile.NamedTemporaryFile('wb')
    pickle.dump(decoder, tf, 2)
    tf.flush()
    database = xmlrpc.client.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)
    database.save_bmi(name, int(entry), tf.name)

def cache_and_train(*args, **kwargs):
    """
    Cache plexon file (if using plexon system) and train BMI.
    """
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
    from . import models

    recording_sys = models.KeyValueStore.get('recording_sys', None)
    if recording_sys == 'plexon':
        print("cache and train")
        entry = kwargs['entry']
        print(entry)
        plxfile = models.DataFile.objects.get(system__name='plexon', entry=entry)
        print(plxfile)

        if not plxfile.has_cache():
            cache = cache_plx.si(plxfile.get_path())
            train = make_bmi.si(*args, **kwargs)
            chain(cache, train)()
        else:
            print("calling")
            make_bmi.delay(*args, **kwargs)
    else:
        make_bmi.delay(*args, **kwargs)

def save_new_decoder_from_existing(obj, orig_decoder_record, suffix='_', dbname='default'):
    '''
    Save a decoder that is created by manipulating the parameters of an older decoder

    Parameters
    ----------
    obj: riglib.bmi.Decoder instance
        New decoder object to be saved
    orig_decoder_record: tracker.models.Decoder instance
        Database record of the original decoder
    suffix: string, default='_'
        The name of the new decoder is created by taking the name of the old decoder and adding the specified suffix

    Returns
    -------
    None
    '''
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
    from . import dbq
    import riglib.bmi
    if not isinstance(obj, riglib.bmi.bmi.Decoder):
        raise ValueError("This function is only intended for saving Decoder objects!")

    new_decoder_fname = obj.save()
    new_decoder_name = orig_decoder_record.name + suffix
    training_block_id = orig_decoder_record.entry_id
    print("Saving new decoder:", new_decoder_name)
    dbq.save_bmi(new_decoder_name, training_block_id, new_decoder_fname, dbname=dbname)

## Functions to manipulate existing (KF)Decoders. These belong elsewhere

def conv_mm_dec_to_cm(decoder_record):
    '''
    Convert a mm unit decoder to cm
    '''
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
    from . import dbq
    decoder_fname = os.path.join('/storage/decoders/', decoder_record.path)
    print(decoder_fname)
    decoder_name = decoder_record.name
    dec = pickle.load(open(decoder_fname))
    from riglib.bmi import train
    dec_cm = train.rescale_KFDecoder_units(dec, 10)

    new_decoder_basename = os.path.basename(decoder_fname).rstrip('.pkl') + '_cm.pkl'
    new_decoder_fname = '/tmp/%s' % new_decoder_basename
    pickle.dump(dec_cm, open(new_decoder_fname, 'w'))

    new_decoder_name = decoder_name + '_cm'
    training_block_id = decoder_record.entry_id
    print(new_decoder_name)
    dbq.save_bmi(new_decoder_name, training_block_id, new_decoder_fname)

def zero_out_SSKF_bias(decoder_record):
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'

    dec = open_decoder_from_record(decoder_record)
    dec.filt.C_xpose_Q_inv_C[:,-1] = 0
    dec.filt.C_xpose_Q_inv_C[-1,:] = 0
    save_new_decoder_from_existing(dec, decoder_record, suffix='_zero_bias')

def conv_kfdecoder_binlen(decoder_record, new_binlen):
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
    dec = open_decoder_from_record(decoder_record)
    dec.change_binlen(new_binlen)
    save_new_decoder_from_existing(dec, decoder_record, suffix='_%dHz' % int(1./new_binlen))

def conv_kfdecoder_to_ppfdecoder(decoder_record):
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
    from . import dbq

    # Load the decoder
    decoder_fname = os.path.join('/storage/decoders/', decoder_record.path)
    print(decoder_fname)
    decoder_name = decoder_record.name
    dec = pickle.load(open(decoder_fname))

    from riglib.bmi import train
    dec_ppf = train.convert_KFDecoder_to_PPFDecoder(dec)

    new_decoder_basename = os.path.basename(decoder_fname).rstrip('.pkl') + '_ppf.pkl'
    new_decoder_fname = '/tmp/%s' % new_decoder_basename
    pickle.dump(dec_ppf, open(new_decoder_fname, 'w'))

    new_decoder_name = decoder_name + '_ppf'
    training_block_id = decoder_record.entry_id
    print(new_decoder_name)
    from . import dbq
    dbq.save_bmi(new_decoder_name, training_block_id, new_decoder_fname)

def conv_kfdecoder_to_sskfdecoder(decoder_record):
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'

    dec = open_decoder_from_record(decoder_record)

    F, K = dec.filt.get_sskf()
    from riglib.bmi import sskfdecoder
    filt = sskfdecoder.SteadyStateKalmanFilter(F=F, K=K)
    dec_sskf = sskfdecoder.SSKFDecoder(filt, dec.units, dec.ssm, binlen=decoder.binlen)

    save_new_decoder_from_existing(decoder_record, '_sskf')

def make_kfdecoder_interpolate(decoder_record):
    os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
    from . import dbq

    # Load the decoder
    decoder_fname = os.path.join('/storage/decoders/', decoder_record.path)
    print(decoder_fname)
    decoder_name = decoder_record.name
    dec = pickle.load(open(decoder_fname))

    from riglib.bmi import train
    dec_ppf = train._interpolate_KFDecoder_state_between_updates(dec)

    new_decoder_basename = os.path.basename(decoder_fname).rstrip('.pkl') + '_ppf.pkl'
    new_decoder_fname = '/tmp/%s' % new_decoder_basename
    pickle.dump(dec_ppf, open(new_decoder_fname, 'w'))

    new_decoder_name = decoder_name + '_60hz'
    training_block_id = decoder_record.entry_id
    print(new_decoder_name)
    from . import dbq
    dbq.save_bmi(new_decoder_name, training_block_id, new_decoder_fname)

