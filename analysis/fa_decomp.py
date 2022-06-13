import numpy as np
import sklearn.decomposition as skdecomp
from db import dbfunctions as dbfn
import tables
#import test_reconst_bmi_traj as trbt
import matplotlib.pyplot as plt
# import seaborn
# seaborn.set_context('talk',font_scale=1)
# seaborn.set_style('whitegrid')
import pickle
import scipy
import os
# te = 3138

# if te==3136:
#     hdf = tables.openFile('grom_data/grom20150427_02.hdf')
# elif te ==3138:
#     hdf = tables.openFile('grom_data/grom20150428_02.hdf')
#hdf = tables.openFile('grom_data/grom20151129_19_te3725.hdf')
#dec = pickle.load(open('grom_data/grom20150422_13_PM04221738.pkl'))
#dec = pickle.load(open('grom_data/grom20151129_16_RMLC11291807.pkl'))
#hdf = tables.openFile(os.path.expandvars('$FA_GROM_DATA/grom20151201_03_te3729.hdf'))
#dec = pickle.load(open(os.path.expandvars('$FA_GROM_DATA/grom20151201_01_RMLC12011916.pkl')))

#ReDecoder = trbt.RerunDecoding(hdf, dec, task='point_mass', drives_neurons=3)
#ReDecoder = trbt.RerunDecoding(hdf, dec, task='bmi_multi')#

########################
########################
#### ANALYSIS FCNS #####
########################
########################
# def FA_k_500ms(): 
#     #Function to determine optimal number of factors for first 500 ms of trial
#     #Separates FA model for each target
#     rew_ix = get_trials_per_min(hdf)
#     spk, targ_pos, targ_ix, reach_time = extract_trials(hdf, rew_ix, ms=500)
#     LL = fit_all_targs(spk, targ_ix,iters=10, max_k = 10)
hdf = []

def FA_k_ALLms(hdf):
    #Function to determine optimal number of factors for full trial
    #Separates FA model for each target
    drives_neurons_ix0 = 3
    rew_ix = get_trials_per_min(hdf)
    internal_state = hdf.root.task[:]['internal_decoder_state']
    update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, drives_neurons_ix0, 0])))[0]+1

    bin_spk, targ_pos, targ_ix, z, zz = extract_trials_all(hdf, rew_ix, update_bmi_ix=update_bmi_ix)
    LL, ax = fit_all_targs(bin_spk, targ_ix, proc_spks = False, iters=20, max_k = 10, return_ax=True) 
    ax[3,0].set_xlabel('Num Factors')
    ax[3,1].set_xlabel('Num Factors')
    for i in range(4):
        ax[i, 0].set_ylabel('Log Lik')
    return LL, ax

def FA_all_targ_ALLms(hdf, max_k=10, iters=20):
    #Function to determine optimal number of factors for full trial
    drives_neurons_ix0 = 3
    rew_ix = get_trials_per_min(hdf)
    internal_state = hdf.root.task[:]['internal_decoder_state']
    update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, drives_neurons_ix0, 0])))[0]+1
    bin_spk, targ_pos, targ_ix, z, zz = extract_trials_all(hdf, rew_ix, update_bmi_ix=update_bmi_ix)
    zscore_X, mu = zscore_spks(bin_spk)
    log_lik, psv, ax = find_k_FA(zscore_X, iters=iters, max_k = max_k, plot=True)
    return log_lik, ax

def SVR_slow_fast_ALLms(hdf, factors=5):
    rew_ix = get_trials_per_min(hdf)
    bin_spk, targ_pos, targ_ix, trial_ix, reach_time = extract_trials_all(hdf, rew_ix)
    fast_vs_slow_trials(bin_spk, targ_ix, reach_time, factors=factors,plot=True, all_trial_tm=True)

def ind_vs_all_targ_SSAlign(factors=5, hdf=hdf):
    rew_ix = get_trials_per_min(hdf)
    bin_spk, targ_pos, targ_ix, trial_ix, reach_time = extract_trials_all(hdf, rew_ix)
    overlap = targ_vs_all_subspace_align(bin_spk, targ_ix, factors=factors)
    f, ax = plt.subplots()
    c = ax.pcolormesh(overlap)
    plt.colorbar(c)
    return overlap


def learning_curve_metrics(hdf_list, epoch_size=56, n_factors=5):
    #hdf_list = [3822, 3834, 3835, 3840]
    #obstacle learning: hdf_list = [4098, 4100, 4102, 4104, 4114, 4116, 4118, 4119]
    rew_ix_list = []
    te_refs = []
    rpm_list = []
    hdf_dict = {}
    perc_succ = []
    time_list = []
    offs = 0

    #f, ax = plt.subplots()
    for te in hdf_list:
        hdf_t = dbfn.TaskEntry(te)
        hdf = hdf_t.hdf
        hdf_dict[te] = hdf

        rew_ix, rpm = pa.get_trials_per_min(hdf, nmin=2,rew_per_min_cutoff=0, 
            ignore_assist=True, return_rpm=True)
        ix = 0
        #ax.plot(rpm)

        trial_ix = np.array([i for i in hdf.root.task_msgs[:] if 
            i['msg'] in ['reward','timeout_penalty','hold_penalty','obstacle_penalty'] ], dtype=hdf.root.task_msgs.dtype)


        while (ix+epoch_size) < len(rew_ix):
            start_rew_ix = rew_ix[ix]
            end_rew_ix = rew_ix[ix+epoch_size]
            msg_ix_mod = np.nonzero(scipy.logical_and(trial_ix['time']<=end_rew_ix, trial_ix['time']>start_rew_ix))[0]
            all_msg = trial_ix[msg_ix_mod]
            perc_succ.append(len(np.nonzero(all_msg['msg']=='reward')[0]) / float(len(all_msg)))

            rew_ix_list.append(rew_ix[ix:ix+epoch_size])
            rpm_list.append(np.mean(rpm[ix:ix+epoch_size]))
            te_refs.append(te)
            time_list.append((0.5*(start_rew_ix+end_rew_ix))+offs)

            ix += epoch_size
        offs = offs+len(hdf.root.task)

    #For each epoch, fit FA model (stick w/ 5 factors for now):
    ratio = []
    for te, r_ix in zip(te_refs, rew_ix_list):
        print(te, len(r_ix))

        update_bmi_ix = np.nonzero(np.diff(np.squeeze(hdf.root.task[:]['internal_decoder_state'][:, 3, 0])))[0] + 1
        bin_spk, targ_pos, targ_ix, z, zz = pa.extract_trials_all(hdf_dict[te], r_ix, time_cutoff=1000, update_bmi_ix=update_bmi_ix)
        zscore_X, mu = pa.zscore_spks(bin_spk)
        FA = skdecomp.FactorAnalysis(n_components=n_factors)
        FA.fit(zscore_X)

        #SOT Variance Ratio by target
        #Priv var / mean
        Cov_Priv = np.sum(FA.noise_variance_)
        U = np.mat(FA.components_).T
        Cov_Shar = np.trace(U*U.T)

        ratio.append(Cov_Shar/(Cov_Shar+Cov_Priv))


########################
########################
###### HELPER FCNS #####
########################
########################

#Get behavior
def get_trials_per_min(hdf,nmin=2, rew_per_min_cutoff=0, ignore_assist=False, return_rpm=False, return_per_succ=False):
    '''
    Summary: Getting trials per minute from hdf file
    Input param: hdf: hdf file to use
    Input param: nmin: number of min to use a rectangular window
    Input param: rew_per_min_cutoff: ignore rew_ix after a 
        certain rew_per_min low threshold is passed
    Output param: rew_ix = rewarded indices in hdf file
    '''

    rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])
    tm = np.zeros((np.max(rew_ix)+1))
    tm[rew_ix] += 1
    
    if hasattr(hdf.root.task, 'assist_level'):
        assist_ix = np.nonzero(hdf.root.task[:]['assist_level']==0)[0]
    else:
        assist_ix = np.zeros((len(hdf.root.task)))

    #Each row occurs ~1/60 sec, so: 
    minute = 60*60;
    min_wind = np.ones((nmin*minute))/float(nmin)
    rew_per_min_tmp = np.convolve(min_wind, tm, mode='same')

    #Now smooth please: 
    smooth_wind = np.ones((3*minute))/float(3*minute)
    rew_per_min = pk_convolve(smooth_wind, rew_per_min_tmp)

    if rew_per_min_cutoff > 0:
        ix = np.nonzero(rew_per_min < rew_per_min_cutoff)[0]
        if len(ix)>0:
            cutoff_ix = ix[0]
        else:
            cutoff_ix = rew_ix[-1]
    
    else:
        cutoff_ix = rew_ix[-1]

    if ignore_assist:
        try:
            beg_zer_assist_ix = assist_ix[0]
        except:
            print('No values w/o assist for filename: ', hdf.filename)
            beg_zer_assist_ix = rew_ix[-1]+1
    else:
        beg_zer_assist_ix = 0


    #plt.plot(np.arange(len(tm))/float(minute), rew_per_min)
    ix_final = scipy.logical_and(rew_ix <= cutoff_ix, rew_ix >= beg_zer_assist_ix)
    if return_rpm:
        return rew_ix[ix_final], rew_per_min[rew_ix[ix_final]]
    else:
        return rew_ix[ix_final]

def pk_convolve(window, arr):
    '''
    Summary: Same as NP convolve but no edge effects (edges padded with zeros+start value and zeros+end value )
    Input param: window: window to convolve
    Input param: arr: array (longer than window)
    Output param: 
    '''
    win_len = len(window)
    start = arr[0]
    end = arr[-1]

    arr_pad = np.hstack(( np.zeros((win_len, )) + start, arr, np.zeros((win_len, )) + end ))
    tmp = np.convolve(window, arr_pad, mode='same')
    return tmp[win_len:-win_len]


def extract_trials_all(hdf, rew_ix, neural_bins = 100, time_cutoff=40, hdf_ix=False, 
    update_bmi_ix=None, rew_pls=False, step_dict=None):
    '''
    Summary: method to extract all time points from trials
    Input param: hdf: task file input
    Input param: rew_ix: rows in the hdf file corresponding to reward times
    Input param: neural_bins: ms per bin
    Input param: time_cutoff: time in minutes, only extract trials before this time
    Input param: hdf_ix: bool, whether to return hdf row corresponding to time of decoder 
    update (and hence end of spike bin)
    Input param: update_bmi_ix: bins for updating bmi

    Input param: rew_plx : If rew_ix is actually a N x 2 array with error trials included (default = false)
    input param: step_dict: Teh number of steps to go backward from each type of trial in rew_ix (only 
        if rew_pls is True)

    Output param: bin_spk -- binned spikes in time x units
                  targ_i_all -- target location at each update
                  targ_ix -- target index 
                  trial_ix -- trial number
                  reach_time -- reach time for trial
                  hdf_ix -- end bin in units of hdf rows
    '''
    if update_bmi_ix is None:
        update_bmi_ix = ReDecoder.update_bmi_ix

    it_cutoff = time_cutoff*60*60

    #Get Go cue and 

    if rew_pls:
        go_ix = np.array([hdf.root.task_msgs[it - step_dict[m['msg']]]['time'] for it, m in enumerate(hdf.root.task_msgs[:]) 
            if m['msg'] in list(step_dict.keys())])

        rew_ix = np.array([hdf.root.task_msgs[it]['time'] for it, m in enumerate(hdf.root.task_msgs[:]) 
            if m['msg'] in list(step_dict.keys())])

        outcome_ix = np.array([hdf.root.task_msgs[it]['msg'] for it, m in enumerate(hdf.root.task_msgs[:]) 
            if m['msg'] in list(step_dict.keys())])

    else:
        go_ix = np.array([hdf.root.task_msgs[it-3][1] for it, t in enumerate(hdf.root.task_msgs[:]) if 
            scipy.logical_and(t[0] == 'reward', t[1] in rew_ix)])
    

    go_ix = go_ix[go_ix<it_cutoff]
    rew_ix = rew_ix[go_ix < it_cutoff]

    targ_i_all = np.array([[-1, -1]])
    trial_ix_all = np.array([-1])
    reach_tm_all = np.array([-1])
    hdf_ix_all = np.array([-1])

    bin_spk = np.zeros((1, hdf.root.task[0]['spike_counts'].shape[0]))-1

    def bin_spks(spk_i, g_ix, r_ix):
        #Need to use 'update_bmi_ix' from ReDecoder to get bin edges correctly:
        trial_inds = np.arange(g_ix, r_ix+1)
        end_bin = np.array([(j,i) for j, i in enumerate(trial_inds) if np.logical_and(i in update_bmi_ix, i>=(g_ix+5))])
        nbins = len(end_bin)
        bin_spk_i = np.zeros((nbins, spk_i.shape[1]))

        hdf_ix_i = []
        for ib, (i_ix, hdf_ix) in enumerate(end_bin):
            #Inclusive of EndBin
            bin_spk_i[ib,:] = np.sum(spk_i[i_ix-5:i_ix+1,:], axis=0)
            hdf_ix_i.append(hdf_ix)
        return bin_spk_i, nbins, np.array(hdf_ix_i)

    for ig, (g, r) in enumerate(zip(go_ix, rew_ix)):
        spk_i = hdf.root.task[g:r]['spike_counts'][:,:,0]

        #Sum spikes in neural_bins:
        bin_spk_i, nbins, hdf_ix_i = bin_spks(spk_i, g, r)
        bin_spk = np.vstack((bin_spk, bin_spk_i))
        targ_i_all = np.vstack(( targ_i_all, np.tile(hdf.root.task[g+1]['target'][[0,2]], (bin_spk_i.shape[0], 1)) ))
        trial_ix_all = np.hstack(( trial_ix_all, np.zeros(( bin_spk_i.shape[0] ))+ig ))
        reach_tm_all = np.hstack((reach_tm_all, np.zeros(( bin_spk_i.shape[0] ))+((r-g)*1000./60.) ))
        hdf_ix_all = np.hstack((hdf_ix_all, hdf_ix_i ))

    print(go_ix.shape, rew_ix.shape, bin_spk.shape, bin_spk_i.shape, nbins, hdf_ix_i.shape)
    targ_ix = get_target_ix(targ_i_all[1:,:])
    
    if hdf_ix:
        return bin_spk[1:,:], targ_i_all[1:,:], targ_ix, trial_ix_all[1:], reach_tm_all[1:], hdf_ix_all[1:]
    else:
        return bin_spk[1:,:], targ_i_all[1:,:], targ_ix, trial_ix_all[1:], reach_tm_all[1:]

def extract_trials(hdf, rew_ix, ms=500, time_cutoff=40):
    it_cutoff = time_cutoff*60*60
    nsteps = int(ms/(1000.)*60)

    #Get Go cue and 
    go_ix = np.array([hdf.root.task_msgs[it-3][1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0] == 'reward'])
    go_ix = go_ix[go_ix<it_cutoff]

    spk = np.zeros((len(go_ix), nsteps, hdf.root.task[0]['spike_counts'].shape[0]))

    targ_pos = np.zeros((len(go_ix), 2))
    reach_time = np.zeros((len(go_ix), ))
    for ig, g in enumerate(go_ix):
        spk[ig, :, :] = hdf.root.task[g:g+nsteps]['spike_counts'][:,:,0]
        targ_pos[ig, :] = hdf.root.task[g+1]['target'][[0,2]]
        reach_time[ig] = (rew_ix[ig]- g)/(60.) #In seconds
    
    targ_ix = get_target_ix(targ_pos)
    return spk, targ_pos, targ_ix, reach_time

def get_target_ix(targ_pos):
    ''' Input: targ_pos - n x 2 array
        Output: n x 1 array of indices
    '''
    #Target Index: 
    b = np.ascontiguousarray(targ_pos).view(np.dtype((np.void, targ_pos.dtype.itemsize * targ_pos.shape[1])))
    _, idx = np.unique(b, return_index=True)
    unique_targ = targ_pos[idx,:]

    #Order by theta: 
    theta = np.arctan2(unique_targ[:,1],unique_targ[:,0])
    thet_i = np.argsort(theta)
    unique_targ = unique_targ[thet_i, :]
    
    targ_ix = np.zeros((targ_pos.shape[0]), )
    for ig, (x,y) in enumerate(targ_pos):
        targ_ix[ig] = np.nonzero(np.sum(targ_pos[ig,:]==unique_targ, axis=1)==2)[0]
    return targ_ix

def proc_spks(spk, targ_ix, targ_ix_analysis=0, neural_bins = 100, return_unshapedX=False):
    '''
    Summary: processes bin_spikes (time x units)
    Input param: spk: time x units (in binned spikes)
    Input param: targ_ix: array lenth of bins
    Input param: targ_ix_analysis: 
    Input param: neural_bins:
    Input param: return_unshapedX:
    Output param: 
    '''

    if targ_ix_analysis == 'all':
        spk_trunc = spk.copy()
    else:
        ix = np.nonzero(targ_ix==targ_ix_analysis)[0]
        spk_trunc = spk[ix, :, :]

    bin_ix = int(neural_bins/1000.*60.)
    spk_trunc_bin = np.zeros((spk_trunc.shape[0], spk_trunc.shape[1]/bin_ix, spk_trunc.shape[2]))
    for i_start in range(spk_trunc.shape[1]/bin_ix):
        spk_trunc_bin[:, i_start, :] = np.sum(spk_trunc[:, (i_start*bin_ix):(i_start+1)*bin_ix, :], axis=1)
    resh_spk_trunc_bin = spk_trunc_bin.reshape((spk_trunc_bin.shape[0]*spk_trunc_bin.shape[1], spk_trunc_bin.shape[2]))
    if return_unshapedX:
        return spk_trunc_bin
    else:
        return resh_spk_trunc_bin

def zscore_spks(proc_spks):
    '''
    proc_spks in time x neurons
    zscore_X in time x neurons
    '''

    mu = np.tile(np.mean(proc_spks, axis=0), (proc_spks.shape[0], 1))
    zscore_X = proc_spks - mu
    return zscore_X, mu

def find_k_FA(zscore_X, xval_test_perc = .1, iters=100, max_k = 20, plot=True):
    ntrials = zscore_X.shape[0]
    ntrain = ntrials*(1-xval_test_perc)
    log_lik = np.zeros((iters, max_k))
    perc_shar_var = np.zeros((iters, max_k))

    for i in range(iters):
        print('iter: ', i)
        ix = np.random.permutation(ntrials)
        train_ix = ix[:ntrain]
        test_ix = ix[ntrain:]

        for k in range(max_k):
            print('factor n : ', k)
            FA = skdecomp.FactorAnalysis(n_components=k+1)
            FA.fit(zscore_X[train_ix,:])
            LL = np.sum(FA.score(zscore_X[test_ix,:]))
            log_lik[i, k] = LL

            if (k+1) == max_k:
                cov_shar = np.matrix(FA.components_) * np.matrix(FA.components_.T)
                perc_shar_var[i, :] = np.cumsum(np.diag(cov_shar)) 

    log_lik[log_lik<-1e4] = np.nan
    if plot:
        fig, ax = plt.subplots(nrows=2)
        mu = 1/float(log_lik.shape[0])*np.nansum(log_lik,axis=0)
        
        for it in range(log_lik.shape[0]):
            for nf in range(log_lik.shape[1]):
                if np.isnan(log_lik[it, nf]):
                    log_lik[it, nf] = mu[nf]

        fin = np.tile(np.array([perc_shar_var[:,-1]]).T, [1, perc_shar_var.shape[1]])
        perc = perc_shar_var/fin.astype(float)

        ax[0].errorbar(np.arange(1,log_lik.shape[1]+1), np.mean(log_lik,axis=0), yerr=np.std(log_lik,axis=0)/np.sqrt(iters), fmt='o')
        #plt.plot(np.arange(1,log_lik.shape[1]+1), np.mean(log_lik,axis=0),'.-')
        ax[0].set_ylabel('Log Likelihood of Held Out Data')

        ax[1].errorbar(np.arange(1,perc.shape[1]+1), np.mean(perc,axis=0), yerr=np.std(perc,axis=0)/np.sqrt(iters), fmt='o')
        ax[1].set_xlabel('Number of Factors')
        ax[1].set_ylabel('Perc. Shar. Var. ')
        return log_lik, perc_shar_var, ax
    else:

        return log_lik, perc_shar_var

def fit_all_targs(spk, targ_ix, proc_spks=True, iters=10, max_k = 10, return_ax=False):
    LL = dict()
    tg = np.unique(targ_ix)
    fig, ax = plt.subplots(nrows=4, ncols=2)

    for it, t in enumerate(tg):
        if proc_spks:
            resh_spk_trunc_bin = proc_spks(spk, targ_ix, targ_ix_analysis=t)
        else:
            ix = np.nonzero(targ_ix==t)[0]
            resh_spk_trunc_bin = spk[ix,:]
            print('ix: ', str(len(ix)))
        zscore_X, mu = zscore_spks(resh_spk_trunc_bin)
        log_lik, psv = find_k_FA(zscore_X, iters=iters, max_k =max_k, plot=False)
        LL[t] = log_lik
        #TODO: insert way to 
        ax[it%4, it/4].plot(np.arange(10)+1, np.mean(log_lik, axis=0), '.-', label='Targ '+str(t))
        ax[it%4, it/4].errorbar(np.arange(1,log_lik.shape[1]+1), np.mean(log_lik,axis=0), yerr=np.std(log_lik,axis=0)/np.sqrt(iters), fmt='o')
        ax[it%4, it/4].set_title('Targ '+str(t))
    
    plt.axis('tight')
    plt.tight_layout()
    if return_ax:
        return LL, ax
    else:
        return LL


def shared_vs_total_var(spk, targ_ix, reach_time, factors=5):
    #For each target, fit a FA analysis model:
    tg = np.unique(targ_ix)
    ratio = np.zeros((len(tg), ))
    beh = ratio.copy()
    for it, t in enumerate(tg):
        zscore_X, z, m = proc_spks(spk, targ_ix, targ_ix_analysis=t)
        FA = skdecomp.FactorAnalysis(n_components=factors)
        FA.fit(zscore_X)

        Cov_Priv = np.sum(FA.noise_variance_)
        U = np.mat(FA.components_).T
        Cov_Shar = np.trace(U*U.T)

        ratio[it] = Cov_Shar/(Cov_Shar+Cov_Priv)

        ix = np.nonzero(targ_ix==t)[0]
        beh[it] = np.mean(reach_time[ix])

    f, ax1 = plt.subplots()
    ax1.plot(ratio, label='Ratio')
    ax1.set_ylabel('Shared to Total Variance')
    ax2 = ax1.twinx()
    ax2.plot(beh, 'k-', label='Reach Time (sec)')
    ax2.set_ylabel('Reach Time')
    ax2.set_title('N Factors: '+str(factors))
    ax1.set_xlabel('Target Number')
    plt.legend()
    return ratio, beh

def fast_vs_slow_trials(spk, targ_ix, reach_time, factors=5,plot=True, all_trial_tm=False):
    tg = np.unique(targ_ix)

    fast_ratio = np.zeros((len(tg), ))
    slow_ratio = fast_ratio.copy()
    beh_fast = fast_ratio.copy()
    beh_slow = fast_ratio.copy()

    for it, t in enumerate(tg):
        ix = np.nonzero(targ_ix==t)[0]
        mid = np.percentile(reach_time[ix], 50)
        fast_ix_ix = np.nonzero(reach_time[ix]<= mid)[0]
        slow_ix_ix = np.nonzero(reach_time[ix]> mid)[0]

        #Fast FA: 
        def get_ratio(ix_ix):
            if all_trial_tm:
                X = spk[ix[ix_ix], :]
                zscore_X, mu = zscore_spks(X)
            else:
                X = proc_spks(spk, targ_ix, targ_ix_analysis=t,return_unshapedX=True)
                X = X[ix_ix, :, :]
                zscore_X = zscore_spks(X.reshape((X.shape[0]*X.shape[1], X.shape[2])))
                
            FA = skdecomp.FactorAnalysis(n_components=factors)
            FA.fit(zscore_X)

            Cov_Priv = np.sum(FA.noise_variance_)
            U = np.mat(FA.components_).T
            Cov_Shar = np.trace(U*U.T)
            return Cov_Shar/(Cov_Shar+Cov_Priv)
        
        fast_ratio[it] = get_ratio(fast_ix_ix)
        slow_ratio[it] = get_ratio(slow_ix_ix)
        beh_fast[it] = np.mean(reach_time[ix][fast_ix_ix])
        beh_slow[it] = np.mean(reach_time[ix][slow_ix_ix])

    if plot:
        f, ax1 = plt.subplots(nrows=4, ncols=2)
        for it in range(len(tg)):
            ii = it%4
            jj = it/4

            ax1[ii, jj].plot(tg[it]+np.array([-.33, .33]), [fast_ratio[it], slow_ratio[it]], 'r*-')
            #ax1[ii, jj].plot(tg[it]+.33, slow_ratio[it], 'b*')
            ax1[ii, jj].set_ylabel('Var. Ratio')
            #ax1[ii, jj].set_ylim([.35, .55])
            ax2 = ax1[ii, jj].twinx()
            ax2.plot(tg[it]+np.array([-.33, .33]), [beh_fast[it], beh_slow[it]], 'bo-', label='Reach Time (sec)')
            #ax2.plot(tg[it]+.33, beh_slow[it], 'bo', label='Slow Reach Time (sec)')

            ax2.set_ylabel('Reach Time')
            ax2.set_title('N Factors: '+str(factors))
            #ax2.set_ylim([2.5, 4.5])
            ax1[ii, jj].set_xlabel('Target Number '+str(tg[it]))
            #plt.legend()
    plt.tight_layout()
    return fast_ratio, slow_ratio, beh_fast, beh_slow

def FA_subspace_align(FA1, FA2):
    U_A = np.mat(FA1.components_.T)
    U_B = np.mat(FA2.components_).T
    v, s, vt = np.linalg.svd(U_B*U_B.T)
    P_B = v*vt
    S_A_shared = U_A*U_A.T
    return np.trace(P_B*S_A_shared*P_B.T)/np.trace(S_A_shared)

def targ_vs_all_subspace_align(bin_spk, targ_ix, factors=5):
    X_zsc, mu = zscore_spks(bin_spk)
    FA_full = skdecomp.FactorAnalysis(n_components = factors)
    FA_full.fit(X_zsc)

    unique_targ = np.unique(targ_ix)
    Overlap = np.zeros((len(unique_targ)+1, len(unique_targ)+1))
    
    FA_targ = dict()
    for it, t in enumerate(unique_targ):
        FA = skdecomp.FactorAnalysis(n_components = factors)
        ix = np.nonzero(targ_ix==t)[0]
        X_zsc, mu = zscore_spks(bin_spk[ix,:])
        FA.fit(X_zsc)
        FA_targ[t] = FA
        Overlap[it, len(unique_targ)] = FA_subspace_align(FA, FA_full)
        Overlap[len(unique_targ), it] = FA_subspace_align(FA_full, FA)

    Overlap[len(unique_targ), len(unique_targ)] = FA_subspace_align(FA_full, FA_full)
        
    for it, t in enumerate(unique_targ):
        for iitt, tt in enumerate(unique_targ):
            Overlap[it, iitt] = FA_subspace_align(FA_targ[t], FA_targ[tt])

    return Overlap




