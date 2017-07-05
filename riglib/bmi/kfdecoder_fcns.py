
from db.tracker import models
from db import dbfunctions as dbfn
from db.tracker.models import Decoder
from db import trainbmi
import numpy as np
import scipy
from kfdecoder import KalmanFilter, KFDecoder

########## MAIN DECODER MANIPULATION METHODS #################

def add_rm_units(task_entry_id, units, add_or_rm, flag_added_for_adaptation, name_suffix='', decoder_entry_id=None):
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

    kfdec = get_decoder_corr(task_entry_id, decoder_entry_id)

    if add_or_rm is 'add':
        kfdec_new , n_new_units = add_units(kfdec, units)
        
        # Only Adapt new units: 
        if flag_added_for_adaptation:
            kfdec_new.adapting_neural_inds = np.zeros((len(kfdec_new.units)))
            kfdec_new.adapting_neural_inds[-1*len(units):] = 1

        save_new_dec(task_entry_id, kfdec_new, name_suffix+'_add_'+str(n_new_units)+'_units')

    elif add_or_rm is 'rm':
        orig_units = kfdec.units
        inds_to_keep = proc_units(kfdec, units, 'remove')
        if len(orig_units) == len(inds_to_keep):
            print ' units cannot be removed since theyre not in original decoder', orig_units
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

def zscore_units(task_entry_id, decoder_entry_id, calc_zscore_from_te):
    
    decoder = get_decoder_corr(task_entry_id, decoder_entry_id)

    # Init mFR / sdFR
    decoder.mFR = 0.
    decoder.sdFR = 1.

    hdf = dbfn.TaskEntry(calc_zscore_from_te).hdf
    spk_counts = hdf.root.task[:]['spike_counts'][:, :, 0]
    
    # Make sure not repeated entries:
    sum_spk_counts = np.sum(spk_counts, axis=1)
    ix = np.nonzero(sum_spk_counts)[0][0]
    sample = 1+ sum_spk_counts[ix:ix+6] - sum_spk_counts[ix]
    assert np.sum(sample) != 6

    mFR = np.squeeze(np.mean(spk_counts, axis=0))
    sdFR = np.std(spk_counts, axis=0)

    decoder.init_zscore(mFR, sdFR)
    save_new_dec(task_entry_id, decoder, '_zscore_set_from_'+str(calc_zscore_from_te))

def adj_state_noise(task_entry_id, decoder_entry_id, new_w):
    decoder = get_decoder_corr(task_entry_id, decoder_entry_id, return_used_te=True)
    W = np.diag(decoder.filt.W)
    wix = np.nonzero(W)[0]
    W[wix] = new_w
    W_new = np.diag(W)
    decoder.filt.W = W_new
    save_new_dec(task_entry_id, decoder, '_W_new_'+str(new_w))    

########################## HELPER DECODER MANIPULATION METHODS #################################

def get_decoder_corr(task_entry_id, decoder_entry_id):
    '''
    Summary: get KF decoder either from entry that has trained the decoder (if this, need decoder_entry_id if > 1 decoder), 
        or decoder that was used during task_entry_id
    Input param: task_entry_id: dbname task entry ID
    Input param: decoder_entry_id: decoder entry id: (models.Decoder.objects.get(entry=entry))
    Output param: KF Decoder
    '''

    decoder_entries = dbfn.TaskEntry(task_entry_id).get_decoders_trained_in_block()
    if len(decoder_entries) > 0:
        print 'Loading decoder TRAINED from task %d'%task_entry_id
        if type(decoder_entries) is models.Decoder:
            decoder = decoder_entries
        else: # list of decoders. Search for the right one. 
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
            print 'Loading decoder USED in task %s'%dbfn.TaskEntry(task_entry_id).task
        except:
            raise Exception('Cannot load decoder from TE%d'%task_entry_id)
    if return_used_te:
        return kfdec, 
    else:
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

def save_new_dec(task_entry_id, dec_obj, suffix):
    '''
    Summary: Method to save decoder to DB -- saves to TE that original decoder came from
    Input param: task_entry_id: original task to save decoder to
    Input param: dec_obj: KF decoder new
    Input param: suffix:
    Output param: 
    '''

    te = dbfn.TaskEntry(task_entry_id)
    try:
        te_id = te.te_id
    except:
        dec_nm = te.name
        te_ix = dec_nm.find('te')
        te_ix_end = dec_nm.find('_',te_ix)
        if te_ix_end == -1:
            te_ix_end = len(dec_nm)
        te_id = int(dec_nm[te_ix+2:te_ix_end])

    old_dec_obj = te.decoder_record
    trainbmi.save_new_decoder_from_existing(dec_obj, old_dec_obj, suffix=suffix)
