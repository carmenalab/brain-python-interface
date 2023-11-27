
from db.tracker import models
from db import dbfunctions as dbfn
from db.tracker import trainbmi
import numpy as np
from .kfdecoder import KalmanFilter, KFDecoder
from . import train
import pickle
import re
import tables
import os

########## MAIN DECODER MANIPULATION METHODS #################

def add_rm_units(task_entry_id, units, add_or_rm, flag_added_for_adaptation, name_suffix='', decoder_entry_id=None, **kwargs):
    '''
    Summary: Method to add or remove units from KF decoder. 
        Takes in task_entry_id number or decoder_entry_id to get decoder
        Removes or adds units to decoder
            If adds, sets decoder weights to random entries


    Input param: task_entry_id: Decoder = dbfn.TaskEntry(task_entry_id).get_decoders_trained_in_block()
    Input param: units: list of units to add or remove
    Input param: add_or_rm: 'add' or 'rm' 
    Input param: flag_added_for_adaptation: whether or not to flag newly added units as adapting inds
    Input param: name_suffix: new decoder suffix. If empty, will append 'add_or_rm_units_len(units)'
    Input param: decoder_entry_id: used if more than 1 decoder training on block

    '''
    if 'decoder_path' in kwargs:
        kfdec = pickle.load(open(kwargs['decoder_path']))
    else:
        kfdec = get_decoder_corr(task_entry_id, decoder_entry_id)

    if add_or_rm is 'add':
        kfdec_new , n_new_units = add_units(kfdec, units)
        
        # Only Adapt new units: 
        if flag_added_for_adaptation:
            kfdec_new.adapting_neural_inds = np.arange(len(kfdec_new.units)-len(units), len(kfdec_new.units))

        save_new_dec(task_entry_id, kfdec_new, name_suffix+'_add_'+str(n_new_units)+'_units')

    elif add_or_rm is 'rm':
        orig_units = kfdec.units
        inds_to_keep = proc_units(kfdec, units, 'remove')
        if len(orig_units) == len(inds_to_keep):
            print(' units cannot be removed since theyre not in original decoder', orig_units)
        else:
            dec_new = return_proc_units_decoder(kfdec, inds_to_keep)
            save_new_dec(task_entry_id, dec_new, name_suffix+'_rm_'+str(len(units))+'_units')

def flag_adapting_inds_for_CLDA(task_entry_id, state_names_to_adapt=None, units_to_adapt = None, decoder_entry_id=None):
    
    decoder = get_decoder_corr(task_entry_id, decoder_entry_id)
    state_adapting_inds = []
    for s in state_names:
        ix = np.nonzero(decoder.states == s)[0]
        assert len(ix) == 0
        state_adapting_inds.append(int(ix))
    decoder.adapting_state_inds = np.array(adapting_inds)

    neural_adapting_inds = []
    for u in units_to_adapt:
        uix = np.nonzero(np.logical_and(decoder.units[:, 0]== u[0], decoder.units[:, 1]== u[1]))[0]
        neural_adapting_inds.append(int(uix))
    decoder.adapting_neural_inds = np.array(neural_adapting_inds)
    save_new_dec(task_entry_id, decoder, '_adapt_only_'+str(len(units_to_adapt))+'_units_'+str(len(state_names_to_adapt))+'_states')

def zscore_units(task_entry_id, calc_zscore_from_te, pos_key = 'cursor', decoder_entry_id=None, 
    training_method=train.train_KFDecoder, retrain_flag = False, **kwargs):
    '''
    Summary: Method to be able to 'convert' a trained decoder (that uses zscoring) to one that uses z-scored from another session
         (e.g. you train a decoder from VFB, but you want to zscore unit according to a passive / still session earlier). You 
         would use the task_entry_id that was used to train the decoder OR entry that used the decoder. Then 'calc_zscore_from_te'
         is the task entry ID used to compute the z-scored units. You can either retrain the decoder iwth the new z-scored units, or not

    Input param: task_entry_id:
    Input param: decoder_entry_id:
    Input param: calc_zscore_from_te:
    Output param: 
    '''
    if 'decoder_path' in kwargs:
        decoder = pickle.load(open(kwargs['decoder_path']))
    else:
        decoder = get_decoder_corr(task_entry_id, decoder_entry_id, get_dec_used=False)

    assert (hasattr(decoder, 'zscore') and decoder.zscore is True)," Cannot update mFR /sdFR of decoder that was not trained as zscored decoder. Retrain!"

    # Init mFR / sdFR
    if 'hdf_path' in kwargs:
        hdf = tables.openFile(kwargs['hdf_path'])
    else:
        hdf = dbfn.TaskEntry(calc_zscore_from_te).hdf
    
    # Get HDF update rate from hdf file. 
    try:
        hdf_update_rate = np.round(np.mean(hdf.root.task[:]['loop_time'])*1000.)/1000.
    except:
        from config import config
        if config.recording_system == 'blackrock':
            hdf_update_rate = .05;
        elif config.recording_system == 'plexon':
            hdf_update_rate = 1/60.

    spk_counts = hdf.root.task[:]['spike_counts'][:, :, 0]
    
    # Make sure not repeated entries:
    sum_spk_counts = np.sum(spk_counts, axis=1)
    ix = np.nonzero(sum_spk_counts)[0][0]
    sample = 1+ sum_spk_counts[ix:ix+6] - sum_spk_counts[ix]
    assert np.sum(sample) != 6

    decoder_update_rate = decoder.binlen
    bin_spks, _ = bin_(None, spk_counts.T, hdf_update_rate, decoder_update_rate, only_neural=True)
    mFR = np.squeeze(np.mean(bin_spks, axis=1))
    sdFR = np.std(bin_spks, axis=1)
    kwargs2 = dict(mFR=mFR, sdFR=sdFR)

    #Retrain decoder w/ new zscoring:
    if 'te_id' in kwargs:
        training_id = kwargs['te_id']
    else:
        training_id = decoder.te_id
    

    if retrain_flag:
        raise NotImplementedError("Need to test retraining with real data")
        saved_files = models.DataFile.objects.filter(entry_id=training_id)
        files = {}
        for fl in saved_files:
            files[fl.system.name] = fl.get_path()
        import bmilist
        decoder = training_method(files, decoder.extractor_cls, decoder.extractor_kwargs, bmilist.kin_extractors[''], decoder.ssm, 
            decoder.units, update_rate=decoder.binlen, tslice=decoder.tslice, pos_key=pos_key, zscore=True, **kwargs2)
        suffx = '_zscore_set_from_'+str(calc_zscore_from_te)+'_retrained'
    else:
        decoder.mFR = 0.
        decoder.sdFR = 1.
        decoder.init_zscore(mFR, sdFR)
        decoder.mFR = mFR
        decoder.sdFR = sdFR

        suffx = '_zscore_set_from_'+str(calc_zscore_from_te)

    if task_entry_id is not None:
        save_new_dec(task_entry_id, decoder, suffx)
    else:
        return decoder, suffx

def zscore_features(task_entry_id, calc_zscore_from_te, decoder_entry_id=None, kin_source = 'cursor', **kwargs):
    '''
    Summary: Method to be able to 'convert' a trained decoder (that uses zscoring) to one that uses z-scored from another session
         (e.g. you train a decoder from VFB, but you want to zscore unit according to a passive / still session earlier). You 
         would use the task_entry_id that was used to train the decoder OR entry that used the decoder. Then 'calc_zscore_from_te'
         is the task entry ID used to compute the z-scored features. Unlike zscore_units, this recomputes the features using the 
         feature extractor and extractor_kwargs stored in the decoder.

    Input param: task_entry_id:
    Input param: decoder_entry_id:
    Input param: calc_zscore_from_te:
    Output param: 
    '''
    if 'decoder_path' in kwargs:
        decoder = pickle.load(open(kwargs['decoder_path']))
    else:
        decoder = get_decoder_corr(task_entry_id, decoder_entry_id, get_dec_used=False)

    # assert (hasattr(decoder, 'zscore') and decoder.zscore is True)," Cannot update mFR /sdFR of decoder that was not trained as zscored decoder. Retrain!"

    # Extract new features
    te = models.TaskEntry.objects.get(id=calc_zscore_from_te)
    te_json = te.to_json()
    files = te.get_data_files_dict_absolute()
    binlen = decoder.binlen
    units = decoder.units
    extractor_cls = decoder.extractor_cls
    extractor_kwargs = decoder.extractor_kwargs

    # Load the recording to get the length
    recording_sys_make = models.KeyValueStore.get('recording_sys')
    if recording_sys_make == 'plexon':
        from plexon import plexfile # keep this import here so that only plexon rigs need the plexfile module installed
        plexon = models.System.objects.get(name='plexon')
        df = models.DataFile.objects.get(entry=te.id, system=plexon)
        plx = plexfile.openFile(df.get_path().encode('utf-8'), load=False)
        length = plx.length
    elif recording_sys_make == 'ecube':
        sys = models.System.objects.get(name='ecube')
        df = models.DataFile.objects.get(entry=te.id, system=sys)
        filepath = df.get_path()
        from riglib.ecube import parse_file
        info = parse_file(str(df.get_path()))
        length = info.length
    else:
        ValueError('Unrecognized recording_system!')

    tslice = [0, length]
    strobe_rate = te.task_params['fps']

    neural_features, units, extractor_kwargs = train.get_neural_features(files, binlen, extractor_cls.extract_from_file,
        extractor_kwargs, tslice=tslice, units=units, source=kin_source, strobe_rate=strobe_rate)
    mFR = np.squeeze(np.mean(neural_features, axis=0))
    sdFR = np.std(neural_features, axis=0)

    # Update decoder w/ new zscoring:   
    decoder.mFR = 0.
    decoder.sdFR = 1.
    decoder.init_zscore(mFR, sdFR)
    decoder.mFR = mFR
    decoder.sdFR = sdFR

    # Save it as a new decoder
    suffx = '_zscore_set_from_'+str(calc_zscore_from_te)

    if task_entry_id is not None:
        save_new_dec(task_entry_id, decoder, suffx)
    else:
        return decoder, suffx



def adj_state_noise(task_entry_id, decoder_entry_id, new_w):
    decoder = get_decoder_corr(task_entry_id, decoder_entry_id, return_used_te=True)
    W = np.diag(decoder.filt.W)
    wix = np.nonzero(W)[0]
    W[wix] = new_w
    W_new = np.diag(W)
    decoder.filt.W = W_new
    save_new_dec(task_entry_id, decoder, '_W_new_'+str(new_w))    

########################## HELPER DECODER MANIPULATION METHODS #################################

def get_decoder_corr(task_entry_id, decoder_entry_id, get_dec_used=True):
    '''
    Summary: get KF decoder either from entry that has trained the decoder (if this, need decoder_entry_id if > 1 decoder), 
        or decoder that was used during task_entry_id
    Input param: task_entry_id: dbname task entry ID
    Input param: decoder_entry_id: decoder entry id: (models.Decoder.objects.get(entry=entry))
    Output param: KF Decoder
    '''
    ld = True
    if get_dec_used is False:
        decoder_entries = dbfn.TaskEntry(task_entry_id).get_decoders_trained_in_block()
        if type(decoder_entries) is models.Decoder:
            decoder = decoder_entries
            ld = False
        else: # list of decoders. Search for the right one. 
            try:
                dec_ids = [de.pk for de in decoder_entries]
                _ix = np.where(np.isin(dec_ids, decoder_entry_id))[0][0]
                decoder = decoder_entries[_ix]
                ld = False
            except:
                if decoder_entry_id is None:
                    print('Too many decoder entries trained from this TE, specify decoder_entry_id')
                else:
                    print('Too many decoder entries trained from this TE, no match to decoder_entry_id %d'%decoder_entry_id)
    if ld is False:
        kfdec = decoder.load()            
    else:
        try:
            kfdec = dbfn.TaskEntry(task_entry_id).decoder
            print('Loading decoder USED in task %s'%dbfn.TaskEntry(task_entry_id).task)
        except:
            raise Exception('Cannot load decoder from TE%d'%task_entry_id)
    return kfdec

def add_units(kfdec, units):
    '''
    Add units to KFDecoder, e.g. to account for appearance of new cells 
    on a particular day, will need to do CLDA to fit new deocder weight
    
    Parameters: 
    units: string or np.ndarray of shape (N, 2) of units to ADD to current decoder
    '''
    units_curr = kfdec.units
    new_units = proc_units(kfdec, units, 'to_int')

    keep_ix = []
    for r, r_un in enumerate(new_units):
        if len(np.nonzero(np.all(r_un==units_curr, axis=1))[0]) > 0: 
            print('not adding unit ', r_un, ' -- already in decoder')
        else:
            keep_ix.append(r)

    new_units = np.array(new_units)[keep_ix, :]
    units = np.vstack((units_curr, new_units))
    n_states = kfdec.filt.C.shape[1]
    n_features = len(units)

    C = np.vstack(( kfdec.filt.C, 1e-3*np.random.randn(len(new_units), kfdec.ssm.n_states)))
    Q = np.eye( len(units), len(units) )
    Q[np.ix_(np.arange(len(units_curr)), np.arange(len(units_curr)))] = kfdec.filt.Q
    Q_inv = np.linalg.inv(Q)

    if isinstance(kfdec.mFR, np.ndarray):
        mFR = np.hstack(( kfdec.mFR, np.zeros((len(new_units))) ))
        sdFR = np.hstack(( kfdec.sdFR, np.ones((len(new_units))) ))
    else:
        mFR = kfdec.mFR
        sdFR = kfdec.sdFR

    filt = KalmanFilter(A=kfdec.filt.A, W=kfdec.filt.W, C=C, Q=Q, is_stochastic=kfdec.filt.is_stochastic)
    C_xpose_Q_inv = C.T * Q_inv
    C_xpose_Q_inv_C = C.T * Q_inv * C
    filt.C_xpose_Q_inv = C_xpose_Q_inv
    filt.C_xpose_Q_inv_C = C_xpose_Q_inv_C        

    filt.R = kfdec.filt.R
    ix = np.random.permutation(n_features)[:len(new_units)]
    filt.S = np.vstack(( kfdec.filt.S, kfdec.filt.S[ix, :]))
    filt.T = Q + filt.S * filt.S.T
    filt.ESS = kfdec.filt.ESS

    decoder = KFDecoder(filt, units, kfdec.ssm, mFR=mFR, sdFR=sdFR, binlen=kfdec.binlen, tslice=kfdec.tslice)
    decoder.n_features = units.shape[0]
    decoder.units = units
    decoder.extractor_cls = kfdec.extractor_cls
    decoder.extractor_kwargs = kfdec.extractor_kwargs
    try:
        CE = kfdec.corresp_encoder
        CE.C = np.vstack((CE.C, 3*np.random.randn(len(new_units), CE.C.shape[1])))
        Q = .1*np.eye(len(units))
        Q[:len(units_curr), :len(units_curr)] = CE.Q
        CE.Q = Q
        CE.n_features = len(units)
        decoder.corresp_encoder = CE
        print('adjusted corresp_encoder too!')
    except:
        pass
    decoder.extractor_kwargs['units'] = units
    return decoder, len(keep_ix)

def proc_units(kfdec, units, mode):
    '''
    Parse list of units indices to keep from string or np.ndarray of shape (N, 2)
    Inputs: 
        units -- 
        mode -- can be 'keep' or 'remove' or 'to_int'. Tells function what to do with the units
    '''

    if isinstance(units[0], str):
        # convert to array
        if isinstance(units, str):
            units = units.split(', ')

        units_lut = dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10, k=11)
        units_int = []
        for u in units:
            ch = int(re.match('(\d+)([a-k])', u).group(1))
            unit_ind = re.match('(\d+)([a-k])', u).group(2)
            # import pdb; pdb.set_trace()
            units_int.append((ch, units_lut[unit_ind]))

        units = units_int
    
    if mode == 'to_int':
        return units

    inds_to_keep = []
    new_units = list(map(tuple, units))
    for k, old_unit in enumerate(kfdec.units):
        if mode == 'keep':
            if tuple(old_unit) in new_units:
                inds_to_keep.append(k)
        elif mode == 'remove':
            if tuple(old_unit) not in new_units:
                inds_to_keep.append(k)
    return inds_to_keep

def return_proc_units_decoder(kfdec, inds_to_keep):
    A = kfdec.filt.A
    W = kfdec.filt.W
    C = kfdec.filt.C
    Q = kfdec.filt.Q
    print('Indices to keep: ', inds_to_keep)
    C = C[inds_to_keep, :]
    Q = Q[np.ix_(inds_to_keep, inds_to_keep)]
    Q_inv = np.linalg.inv(Q)

    if isinstance(kfdec.mFR, np.matrix):
        mFR = np.squeeze(np.array(kfdec.mFR))[inds_to_keep]
        sdFR = np.squeeze(np.array(kfdec.sdFR))[inds_to_keep]
        
    elif isinstance(kfdec.mFR, np.ndarray):
        mFR = kfdec.mFR[inds_to_keep]
        sdFR = kfdec.mFR[inds_to_keep]
    else:
        mFR = kfdec.mFR
        sdFR = kfdec.sdFR

    filt = KalmanFilter(A=A, W=W, C=C, Q=Q, is_stochastic=kfdec.filt.is_stochastic)
    C_xpose_Q_inv = C.T * Q_inv
    C_xpose_Q_inv_C = C.T * Q_inv * C
    filt.C_xpose_Q_inv = C_xpose_Q_inv
    filt.C_xpose_Q_inv_C = C_xpose_Q_inv_C        

    units = kfdec.units[inds_to_keep]

    filt.R = kfdec.filt.R
    filt.S = kfdec.filt.S[inds_to_keep, :]
    filt.T = kfdec.filt.T[np.ix_(inds_to_keep, inds_to_keep)]
    filt.ESS = kfdec.filt.ESS

    decoder = KFDecoder(filt, units, kfdec.ssm, mFR=mFR, sdFR=sdFR, binlen=kfdec.binlen, tslice=kfdec.tslice)

    decoder.n_features = units.shape[0]
    decoder.units = units
    decoder.extractor_cls = kfdec.extractor_cls
    decoder.extractor_kwargs = kfdec.extractor_kwargs
    decoder.extractor_kwargs['units'] = units
    try:
        CE = kfdec.corresp_encoder
        CE.C = CE.C[inds_to_keep, :]
        CE.Q = CE.Q[np.ix_(inds_to_keep, inds_to_keep)]
        CE.n_features = len(units)
        decoder.corresp_encoder = CE
        print('adjusted corresp_encoder too!')
    except:
        pass

    return decoder

def save_new_dec(task_entry_id, dec_obj, suffix, dbname='default'):
    '''
    Summary: Method to save decoder to DB -- saves to TE that original decoder came from
    Input param: task_entry_id: original task to save decoder to
    Input param: dec_obj: KF decoder new
    Input param: suffix:
    Output param: 
    '''

    te = dbfn.TaskEntry(task_entry_id)
    try:
        te_id = te.id
    except:
        dec_nm = te.name
        te_ix = re.search('te[0-9]',dec_nm)
        ix = te_ix.start() + 2
        sub_dec_nm = dec_nm[ix:]
        
        te_ix_end = sub_dec_nm.find('_')
        if te_ix_end == -1:
            te_ix_end = len(sub_dec_nm)
        te_id = int(sub_dec_nm[:te_ix_end])

    old_dec_obj = te.decoder_record
    if old_dec_obj is None:
        old_dec_obj = faux_decoder_obj(task_entry_id)
    trainbmi.save_new_decoder_from_existing(dec_obj, old_dec_obj, suffix=suffix, dbname=dbname)

def bin_(kin, neural_features, update_rate, desired_update_rate, only_neural=False):

    n = desired_update_rate/float(update_rate)
    if not only_neural:
        assert kin.shape[1] == neural_features.shape[1]
    
    ix_end = int(np.floor(neural_features.shape[1] / n)*n)

    if (n - round(n)) < 1e-5:
        n = int(n)
        if not only_neural:
            kin_ = kin[:, :ix_end].reshape(kin[:, :ix_end].shape[0], kin[:, :ix_end].shape[1]/n, n)
            bin_kf = np.mean(kin_, axis=2)

        nf_ = neural_features[:, :ix_end].reshape(neural_features[:, :ix_end].shape[0], neural_features[:, :ix_end].shape[1]/n, n)
        bin_nf = np.sum(nf_, axis=2)

        if only_neural:
            return bin_nf, desired_update_rate
        else:
            return bin_nf, bin_kf, desired_update_rate
    else:
        raise Exception('Desired rate %f not multiple of original rate %f', desired_update_rate, update_rate)

class faux_decoder_obj(object):
    def __init__(self, task_entry_id, *args,**kwargs):
        self.name = ''
        self.entry_id = task_entry_id
