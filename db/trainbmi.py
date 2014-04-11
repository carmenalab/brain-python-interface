'''
Functions to train decoders and store into the database
'''

import re
import cPickle
import tempfile
import xmlrpclib
import pickle
import os

import namelist
from tracker import models
from plexon import plexfile

cellname = re.compile(r'(\d{1,3})\s*(\w{1})')

from celery import task, chain
from tracker import dbq

import numpy as np
from riglib.bmi import extractor

@task()
def cache_plx(plxfile):
    """Create cache for plexon file"""
    plx = plexfile.openFile(str(plxfile)) 

@task()
def make_bmi(name, clsname, extractorname, entry, cells, channels, binlen, tslice):
    """Train BMI

    (see doc for cache_and_train for input argument info)
    """
    extractor_cls = namelist.extractors[extractorname]
    print 'Training with extractor class:', extractor_cls

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
            cells = np.unique(cells)
            units = np.array(cells).astype(np.int32)
    elif ('lfp' in extractor_cls.feature_type) or ('emg' in extractor_cls.feature_type):  # e.g., 'lfp_power'
        # look at "channels" argument (ignore "cells")
        channels = np.array(channels.split(', ')).astype(np.int32)  # convert str to list of numbers
        if len(channels) == 0:
            channels = [1, 2, 3, 4]  # use these channels by default
        else:
            channels = np.unique(channels)
            units = np.hstack([channels.reshape(-1, 1), np.zeros(channels.reshape(-1, 1).shape, dtype=np.int32)])
    else:
        raise Exception('Unknown feature_type!')


    # TODO -- hardcoding this here for now, but eventually the extractor kwargs
    #   should come from a file or from the web interface
    # for LFP extractors, only kwarg that needs to be set here is 'channels'
    #   other kwargs will default to values specified in the class's __init__ 
    extractor_kwargs = dict()
    if extractor_cls == extractor.BinnedSpikeCountsExtractor:
        extractor_kwargs['units'] = units
        extractor_kwargs['n_subbins'] = 1  # TODO -- don't hardcode (not = 1 for PPF!)
    elif extractor_cls == extractor.LFPButterBPFPowerExtractor:
        extractor_kwargs['channels'] = channels
    elif extractor_cls == extractor.LFPMTMPowerExtractor:
        extractor_kwargs['channels'] = channels
    elif extractor_cls == extractor.EMGAmplitudeExtractor:
        extractor_kwargs['channels'] = channels
    else:
        raise Exception("Unknown extractor_cls!")



    database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)

    datafiles = models.DataFile.objects.filter(entry_id=entry)
    inputdata = dict((d.system.name, d.get_path()) for d in datafiles)
    training_method = namelist.bmis[clsname]
    decoder = training_method(extractor_cls, extractor_kwargs, 
                                units=units, binlen=binlen, tslice=tslice, **inputdata)
    tf = tempfile.NamedTemporaryFile('wb')
    cPickle.dump(decoder, tf, 2)
    tf.flush()
    database.save_bmi(name, int(entry), tf.name)

def cache_and_train(name, clsname, extractorname, entry, cells, channels, binlen, tslice):
    """Cache plexon file and train BMI

    Parameters
    ----------
    clsname : string
        BMI algorithm name (passed to namelist lookup table 'bmis')
    extractorname : string
        feature extractor algorithm name (passed to namelist lookup table 'extractors')
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
    """
    plexon = models.System.objects.get(name='plexon')
    plxfile = models.DataFile.objects.get(system=plexon, entry=entry)

    if not plxfile.has_cache():
        cache = cache_plx.si(plxfile.get_path())
        train = make_bmi.si(name, clsname, extractorname, entry, cells, channels, binlen, tslice)
        chain(cache, train)()
    else:
        make_bmi.delay(name, clsname, extractorname, entry, cells, channels, binlen, tslice)

def conv_mm_dec_to_cm(decoder_record):
    '''
    Convert a mm unit decoder to cm
    '''
    decoder_fname = os.path.join('/storage/decoders/', decoder_record.path)
    print decoder_fname
    decoder_name = decoder_record.name
    dec = pickle.load(open(decoder_fname))
    from riglib.bmi import train
    dec_cm = train.rescale_KFDecoder_units(dec, 10)

    new_decoder_basename = os.path.basename(decoder_fname).rstrip('.pkl') + '_cm.pkl'
    new_decoder_fname = '/tmp/%s' % new_decoder_basename
    pickle.dump(dec_cm, open(new_decoder_fname, 'w'))

    new_decoder_name = decoder_name + '_cm'
    training_block_id = decoder_record.entry_id
    print new_decoder_name
    dbq.save_bmi(new_decoder_name, training_block_id, new_decoder_fname)

def open_decoder_from_record(decoder_record):
    decoder_fname = os.path.join('/storage/decoders/', decoder_record.path)
    print decoder_fname
    decoder_name = decoder_record.name
    dec = pickle.load(open(decoder_fname))
    return dec

def zero_out_SSKF_bias(decoder_record):
    dec = open_decoder_from_record(decoder_record)
    dec.filt.C_xpose_Q_inv_C[:,-1] = 0
    dec.filt.C_xpose_Q_inv_C[-1,:] = 0
    save_new_decoder(dec, decoder_record, suffix='_zero_bias')


def save_new_decoder(obj, orig_decoder_record, suffix='_'):
    decoder_fname = os.path.join('/storage/decoders/', orig_decoder_record.path)
    new_decoder_basename = os.path.basename(decoder_fname).rstrip('.pkl') + '%s.pkl' % suffix
    new_decoder_fname = '/tmp/%s' % new_decoder_basename
    pickle.dump(obj, open(new_decoder_fname, 'w'))

    decoder_name = orig_decoder_record.name
    new_decoder_name = decoder_name + suffix
    training_block_id = orig_decoder_record.entry_id
    print new_decoder_name
    dbq.save_bmi(new_decoder_name, training_block_id, new_decoder_fname)

def conv_kfdecoder_binlen(decoder_record, new_binlen):
    dec = open_decoder_from_record(decoder_record)
    dec.change_binlen(new_binlen)
    save_new_decoder(dec, decoder_record, suffix='_%dHz' % int(1./new_binlen))

def conv_kfdecoder_to_ppfdecoder(decoder_record):
    # Load the decoder
    decoder_fname = os.path.join('/storage/decoders/', decoder_record.path)
    print decoder_fname
    decoder_name = decoder_record.name
    dec = pickle.load(open(decoder_fname))

    from riglib.bmi import train
    dec_ppf = train.convert_KFDecoder_to_PPFDecoder(dec)

    new_decoder_basename = os.path.basename(decoder_fname).rstrip('.pkl') + '_ppf.pkl'
    new_decoder_fname = '/tmp/%s' % new_decoder_basename
    pickle.dump(dec_ppf, open(new_decoder_fname, 'w'))

    new_decoder_name = decoder_name + '_ppf'
    training_block_id = decoder_record.entry_id
    print new_decoder_name
    from tracker import dbq
    dbq.save_bmi(new_decoder_name, training_block_id, new_decoder_fname)

def conv_kfdecoder_to_sskfdecoder(decoder_record):
    # Load the decoder
    decoder_fname = os.path.join('/storage/decoders/', decoder_record.path)
    print decoder_fname
    decoder_name = decoder_record.name
    dec = pickle.load(open(decoder_fname))

    F, K = dec.filt.get_sskf()
    from riglib.bmi import sskfdecoder 
    filt = sskfdecoder.SteadyStateKalmanFilter(F=F, K=K)
    dec_sskf = sskfdecoder.SSKFDecoder()

    new_decoder_basename = os.path.basename(decoder_fname).rstrip('.pkl') + '_ppf.pkl'
    new_decoder_fname = '/tmp/%s' % new_decoder_basename
    pickle.dump(dec_ppf, open(new_decoder_fname, 'w'))


    new_decoder_name = decoder_name + '_sskf'




def make_kfdecoder_interpolate(decoder_record):
    # Load the decoder
    decoder_fname = os.path.join('/storage/decoders/', decoder_record.path)
    print decoder_fname
    decoder_name = decoder_record.name
    dec = pickle.load(open(decoder_fname))

    from riglib.bmi import train
    dec_ppf = train._interpolate_KFDecoder_state_between_updates(dec)

    new_decoder_basename = os.path.basename(decoder_fname).rstrip('.pkl') + '_ppf.pkl'
    new_decoder_fname = '/tmp/%s' % new_decoder_basename
    pickle.dump(dec_ppf, open(new_decoder_fname, 'w'))

    new_decoder_name = decoder_name + '_60hz'
    training_block_id = decoder_record.entry_id
    print new_decoder_name
    from tracker import dbq
    dbq.save_bmi(new_decoder_name, training_block_id, new_decoder_fname)

def save_decoder(decoder, name, entry):
    tf = tempfile.NamedTemporaryFile('wb')                                     
    cPickle.dump(decoder, tf, 2)                                               
    tf.flush()                                                                 
    database = xmlrpclib.ServerProxy("http://localhost:8000/RPC2/", allow_none=True)
    database.save_bmi(name, int(entry), tf.name)   
