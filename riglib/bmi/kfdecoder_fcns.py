
from db.tracker import models
from db import dbfunctions as dbfn
from db.tracker.models import Decoder
from db import trainbmi
import numpy as np
import scipy
from kfdecoder import KalmanFilter, KFDecoder

def add_rm_units(task_entry_id, units, add_or_rm, name_suffix, flag_added_for_adaptation, decoder_entry_id=None):
    '''
    Summary: Method to add or remove units from KF decoder. 
        Takes in task_entry_id number or decoder_entry_id to get decoder
        Removes or adds units to decoder
            If adds, sets decoder weights to random entries


    Input param: task_entry_id: Decoder = dbfn.TaskEntry(task_entry_id).get_decoders_trained_in_block()
    Input param: units: list of units to add or remove
    Input param: add_or_rm: 'add' or 'rm' 
    Input param: name_suffix: new decoder suffix. If empty, will append 'add_or_rm_units_len(units)'
    Input param: decoder_entry_id: used if more than 1 decoder training on block

    '''

    decoder_entries = dbfn.TaskEntry(task_entry_id).get_decoders_trained_in_block()
    if len(decoder_entries) > 0:
        if type(decoder_entries) is models.Decoder:
            decoder = decoder_entries
        else:
            try:
                dec_ids = [de.pk for de in decoder_entries]
                _ix = np.nonzero(dec_ids==decoder_entry_id)[0]
                decoder = decoder_entries[_ix]
            except:
                if decoder_entry_id is None:
                    raise Exception('Too many decoder entries trained from this TE, specify decoder_entry_id')
                else:
                    raise Exception('Too many decoder entries trained from this TE, no match to decoder_entry_id %d' %decoder_entry_id)
        kfdec = decoder.load()
    else:
        try:
            kfdec = dbfn.TaskEntry(task_entry_id).decoder
            print 'Loading decoder used in task %s'%dbfn.TaskEntry(task_entry_id).task
        except:
            raise Exception('Cannot load decoder from TE%d'%task_entry_id)
    

    if add_or_rm is 'add':
        kfdec_new , n_new_units = add_units(kfdec, units)
        save_new_dec(task_entry_id, decoder, '_add_'+str(n_new_units)+'_units')

    elif add_or_rm is 'rm':
        inds_to_keep = proc_units(kfdec, units, 'remove')
        dec_new = return_proc_units_decoder(inds_to_keep)
        save_new_dec(task_entry_id, dec_new, '_rm_'+str(len(inds_to_keep))+'_units')

    #flag_added_for_adaptation

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
            print 'not adding unit ', r_un, ' -- already in decoder'
        else:
            keep_ix.append(r)

    new_units = np.array(new_units)[keep_ix, :]
    units = np.vstack((units_curr, new_units))

    C = np.vstack(( kfdec.filt.C, np.random.rand(len(new_units), kfdec.ssm.n_states)))
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
    filt.S = np.vstack(( kfdec.filt.S, np.random.rand(len(new_units), kfdec.filt.S.shape[1])))
    filt.T = Q.copy()
    filt.T[np.ix_(np.arange(len(units_curr)), np.arange(len(units_curr)))] = kfdec.filt.T
    filt.ESS = kfdec.filt.ESS

    decoder = KFDecoder(filt, units, kfdec.ssm, mFR=mFR, sdFR=sdFR, binlen=kfdec.binlen, tslice=kfdec.tslice)
    decoder.n_features = units.shape[0]
    decoder.units = units
    decoder.extractor_cls = kfdec.extractor_cls
    decoder.extractor_kwargs = kfdec.extractor_kwargs
    decoder.extractor_kwargs['units'] = units
    return decoder, len(keep_ix)

def proc_units(kfdec, units, mode):
    '''
    Parse list of units indices to keep from string or np.ndarray of shape (N, 2)
    Inputs: 
        units -- 
        mode -- can be 'keep' or 'remove' or 'to_int'. Tells function what to do with the units
    '''

    if isinstance(units[0], (str, unicode)):
        # convert to array
        if isinstance(units, (str, unicode)):
            units = units.split(', ')

        units_lut = dict(a=1, b=2, c=3, d=4)
        units_int = []
        for u in units:
            ch = int(re.match('(\d+)([a-d])', u).group(1))
            unit_ind = re.match('(\d+)([a-d])', u).group(2)
            # import pdb; pdb.set_trace()
            units_int.append((ch, units_lut[unit_ind]))

        units = units_int
    
    if mode == 'to_int':
        return units

    inds_to_keep = []
    new_units = map(tuple, units)
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
    print 'Indices to keep: ', inds_to_keep
    C = C[inds_to_keep, :]
    Q = Q[np.ix_(inds_to_keep, inds_to_keep)]
    Q_inv = np.linalg.inv(Q)

    if isinstance(kfdec.mFR, np.ndarray):
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

    return decoder

def save_new_dec(kfdec, dec_obj, suffix):
    try:
        te_id = kfdec.te_id
    except:
        dec_nm = kfdec.name
        te_ix = dec_nm.find('te')
        te_ix_end = dec_nm.find('_',te_ix)
        te_id = int(dec_nm[te_ix+2:te_ix_end])

    old_dec_obj = Decoder.objects.filter(entry=te_id)
    trainbmi.save_new_decoder_from_existing(dec_obj, old_dec_obj[0], suffix=suffix)



#def zscore_units(decoder_id_number, calc_zscore_from_te=None):

def flag_units_for_CLDA(decoder, units):
    decoder.adapting_neural_inds = units

def flag_state_dimensions_for_CLDA(decoder, state_names):
    adapting_inds = []
    for s in state_names:
        ix = np.nonzero(decoder.states == s)[0]
        assert len(ix) == 0
        adapting_inds.append(int(ix))
    decoder.adapting_state_inds = np.array(adapting_inds)