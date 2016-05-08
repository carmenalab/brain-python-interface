#!/usr/bin/python
"""
Classes to simulate neural activity (spike firing rates) by various methods.
"""
from __future__ import division
import os
import numpy as np

from scipy.io import loadmat

import numpy as np
from numpy.random import poisson, rand
from scipy.io import loadmat, savemat
from itertools import izip


from scipy.integrate import trapz, simps

ts_dtype = [('ts', float), ('chan', np.int32), ('unit', np.int32)]
ts_dtype_new = [('ts', float), ('chan', np.int32), ('unit', np.int32), ('arrival_ts', np.float64)]

############################
##### Gaussian encoder #####
############################
class KalmanEncoder(object):
    '''
    Models a BMI user as someone who, given an intended state x,
    generates a vector of neural features y according to the KF observation
    model equation: y = Cx + q.
    '''
    def __init__(self, ssm, n_features):
        self.ssm = ssm
        self.n_features = n_features

        drives_neurons = ssm.drives_obs
        nX = ssm.n_states

        C = np.random.standard_normal([n_features, nX])
        C[:, ~drives_neurons] = 0
        Q = np.identity(n_features)

        self.C = C
        self.Q = Q

    def __call__(self, intended_state, **kwargs):
        q = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q).reshape(-1, 1)
        neural_features = np.dot(self.C, intended_state.reshape(-1,1)) + q
        return neural_features

    def get_units(self):
        '''
        Return fake indices corresponding to the simulated units, e.g., (1, 1) represents sig001a in the plexon system
        '''
        return np.array([(k,1) for k in range(self.n_features)])


###########################
##### Poisson encoder #####
###########################
class GenericCosEnc(object):
    '''
    Simulate neurons where the firing rate is a linear function of covariates and the rate parameter goes through a Poisson
    '''
    def __init__(self, C, ssm, return_ts=False, DT=0.1, call_ds_rate=6):
        '''
        Constructor for GenericCosEnc

        Parameters
        ----------
        C : np.ndarray of shape (N, K)
            N is the number of simulated neurons, K is the number of covariates driving neuronal activity. 
            The product of C and the hidden state vector x should give the intended spike rates in Hz
        ssm : state_space_models.StateSpace instance
            ARG_DESCR
        return_ts : bool, optional, default=False
            If True, fake timestamps are returned for each spike event in the same format 
            as real spike data would be delivered over the network during a real experiment. 
            If False, a vector of counts is returned instead. Specify True or False depending on 
            which type of feature extractor you're using for your simulated task. 
        DT : float, optional, default=0.1
            Sampling interval to come up with new spike processes
        call_ds_rate : int, optional, default=6
            Calculating DT / call_ds_rate gives the interval between ticks of the main event loop

        Returns
        -------
        GenericCosEnc instance
        '''
        self.C = C
        self.ssm = ssm
        self.n_neurons = C.shape[0]
        self.call_count = 0
        self.call_ds_rate = call_ds_rate
        self.return_ts = return_ts
        self.DT = DT
        self.unit_inds = np.arange(1, self.n_neurons+1)

        #self.unit_inds = nlen(.hstack((np.array([np.arange(1, self.n_neurons+1)]).T, np.ones(self.n_neurons, 1)))

    def get_units(self):
        '''
        Retrieive the identities of the units in the encoder. Only used because units in real experiments have "names"
        '''
        # Just pretend that each unit is the 'a' unit on a separate electrode
        return np.array([(ind, 1) for ind in self.unit_inds])

    def gen_spikes(self, next_state, mode=None):
        """
        Simulate the spikes    
        
        Parameters
        ----------
        next_state : np.array of shape (N, 1)
            The "next state" to be encoded by this population of neurons
        
        Returns
        -------
        time stamps or counts
            Either spike time stamps or a vector of unit spike counts is returned, depending on whether the 'return_ts' attribute is True

        """

        rates = np.dot(self.C, next_state)
        return self.return_spikes(rates, mode=mode)

    def return_spikes(self, rates, mode=None):
        rates[rates < 0] = 0 # Floor firing rates at 0 Hz
        counts = poisson(rates * self.DT)

        if np.logical_or(mode=='ts', np.logical_and(mode is None, self.return_ts)):
            ts = []
            n_neurons = self.n_neurons
            for k, ind in enumerate(self.unit_inds):
                # separate spike counts into individual time-stamps
                n_spikes = int(counts[k])
                fake_time = (self.call_count + 0.5)* 1./60
                if n_spikes > 0:
                    #spike_data = [(fake_time, int(ind/4)+1, ind % 4) for m in range(n_spikes)] 
                    spike_data = [(fake_time, ind, 1) for m in range(n_spikes)] 
                    ts += (spike_data)

            ts = np.array(ts, dtype=ts_dtype)
            return ts
            
        elif np.logical_or(mode=='counts', np.logical_and(mode is None, self.return_ts is False)):
            return counts

    def __call__(self, next_state, mode=None):
        '''
        See CosEnc.__call__ for docs
        '''        
        if self.call_count % self.call_ds_rate == 0:
            ts_data = self.gen_spikes(next_state, mode=mode)

        else:
            if self.return_ts:
                # return an empty list of time stamps
                ts_data = np.array([])
            else:
                # return a vector of 0's
                ts_data = np.zeros(self.n_neurons)

        self.call_count += 1
        return ts_data


class FACosEnc(GenericCosEnc):
    '''
    Simulate neurons where rate is linear function of underlying factor modulation, rate param through Poisson
    '''
    def __init__(self, C, ssm, DT=0.1, call_ds_rate=6,return_ts=False,max_FR=20, **kwargs):

        super(FACosEnc, self).__init__(C, ssm, DT=DT, call_ds_rate=call_ds_rate, return_ts=return_ts)
        
        #self.input_type = ['priv_unt', 'priv_tun', 'shar_unt', 'shar_tun']
        
        #Parse kwargs: 
        self.lambda_spk_update = 1
        self.wt_sources = kwargs.pop('wt_sources', None)
        self.n_facts = kwargs.pop('n_facts', [3, 3])
        self.state_vel_var = kwargs.pop('state_vel_var', 2.7)
        self.r2 = 1./self.state_vel_var
        #Contribute part of these inputs to each neuron (find alpha weightings: \sum_i=1^6 \alpha_i= 1)
        #Right now allocate neurons to each of these: 
        # if self.wt_sources is None:
        #     wt_sources = np.random.rand(len(self.input_type))
        #     tmp = np.sum(wt_sources)

        #     #Make rows add to 1:
        #     self.wt_sources = wt_sources / tmp

        # Establish number factors for tuned / untuned input sources: 
        self.n_tun_factors = self.n_facts[0]
        self.n_unt_factors = self.n_facts[1]

        self.eps = 1e-15
        
        #Establish mapping from kinematics to factors: 
        #Untuned private:
        self.psi_unt = np.random.random_sample((self.n_neurons, 1))
        self.psi_unt_std = 2

        #self.psi_unt = np.random.normal(0, 1, (self.n_neurons, ssm.n_states))

        #Cosine tuned private: TODO: make private
        #self.psi_tun = 20*(np.random.random_sample((self.n_neurons, ssm.n_states))-0.5)

        #Matched to fit KF data
        self.psi_tun = np.random.normal(0, 1/2.5, (self.n_neurons, ssm.n_states))

        #Only velocity tuned:
        self.psi_tun[:, [0, 1, 2, 4, 6]] = 0


        #Mapping from factors to neurons: 
        # Random
        self.U = 2.5*(np.random.random_sample((self.n_neurons, self.n_tun_factors))-0.5)
        self.W = 5.0*(np.random.random_sample((self.n_neurons, self.n_unt_factors))-0.5)

        #Mapping from states to factors:
        #self.V = 2*(np.random.random_sample((self.n_tun_factors, ssm.n_states))-0.5)
        self.v_ = 2*(np.random.random_sample(self.n_tun_factors)-0.5) * (np.sqrt(self.r2))
        self.V = np.zeros((self.n_tun_factors, ssm.n_states))
        self.V[:,3] = self.v_
        self.V[:,5] = self.y2_eq_r2_min_x2(self.v_, self.r2)
        #Matched to make z_i ~ N(0, I)

        self.bin_step_count = -1

    def gen_spikes_old_1(self, next_state, mode=None):
        
        sub_next_state = np.mat(next_state[self.drives_obs_ix]).ravel()
        enc_factors = np.mat(self.enc) * sub_next_state.T

        rates = np.zeros((self.n_neurons, ))
        shar_noise_fact = 20*np.random.rand()

        for i in range(self.n_neurons):
            #Priv noise:
            rates[i] = rates[i] + np.abs(np.random.normal(0, self.Psi[i], 1))

            #Priv tuned: 
            #print self.Gam[i], next_state[self.drives_obs_ix], next_state[self.drives_obs_ix].shape, np.dot(self.Gam[i], next_state[self.drives_obs_ix])
            tmp = np.dot(self.Gam[i], next_state[self.drives_obs_ix])
            tmp = np.max([0, tmp])
            rates[i] = rates[i] + tmp

            #Shar noise:
            rates[i] = rates[i] + self.V[i]*shar_noise_fact

            #Shar tuned: 
            tmp = np.dot(self.U[i,:], enc_factors)
            tmp = np.max([0, tmp])
            rates[i] = rates[i] + tmp
        return self.return_spikes(rates, mode=mode)

    def gen_spikes_old_2(self, next_state, mode=None):
        if len(next_state.shape) == 1:
            next_state = np.array([next_state]).T

        # Private: 
        unt_priv = self.mod_poisson(self.psi_unt)
        tun_priv = self.mod_poisson(np.dot(self.psi_tun, next_state))
        #np.dot(self.psi_tun, next_state).round(0)
        #self.mod_poisson(np.dot(self.psi_tun, next_state))

        task_shar_tun = self.mod_poisson(np.dot(self.U_tun, np.dot(self.V_tun , next_state)))
        anat_shar_tun = np.dot(self.U_tun, self.mod_poisson(np.dot(self.V_tun, next_state)))

        unt_fact = 14*np.random.rand(self.n_unt_factors, 1)
        task_shar_unt = self.mod_poisson(np.dot(self.U_unt, unt_fact))
        anat_shar_unt = np.dot( self.U_unt, self.mod_poisson(unt_fact))

        counts_all = np.hstack(( unt_priv, tun_priv, task_shar_tun, anat_shar_tun, task_shar_unt, anat_shar_unt))
        counts = np.array([np.sum(np.multiply(counts_all, self.wt_sources), axis = 1)]).T

        if np.logical_or(mode=='ts', np.logical_and(mode is None, self.return_ts)):
            ts = []
            n_neurons = self.n_neurons
            for k, ind in enumerate(self.unit_inds):
                # separate spike counts into individual time-stamps
                n_spikes = int(counts[k])
                fake_time = (self.call_count + 0.5)* 1./60
                if n_spikes > 0:
                    #spike_data = [(fake_time, int(ind/4)+1, ind % 4) for m in range(n_spikes)] 
                    spike_data = [(fake_time, ind, 1) for m in range(n_spikes)] 
                    ts += (spike_data)

            ts = np.array(ts, dtype=ts_dtype)
            return ts
            
        elif np.logical_or(mode=='counts', np.logical_and(mode is None, self.return_ts is False)):
            return counts

    def _gen_state(self):
        s = np.random.normal(0, 7, (7, 1))
        s[[1, 4], :] = 0
        s[-1, 0] = 1
        return s

    def gen_spikes(self, next_state, mode=None):
        self.ns_pk = next_state

        self.priv_tun_bins = np.random.poisson(self.lambda_spk_update, self.n_neurons)
        self.priv_unt_bins = np.random.poisson(1, self.n_neurons)

        self.shar_tun_bins = np.random.poisson(1, )
        self.shar_unt_bins = np.random.poisson(1, )

        if len(next_state.shape) == 1:
            next_state = np.array([next_state]).T

        # Private:
        priv_unt = []
        priv_tun = []
        for n in range(self.n_neurons):

            #Private untuned: 
            if self.priv_unt_bins[n] > 0:
                cnt = []
                for z in range(self.priv_unt_bins[n]):
                    psi_unt = np.max([np.random.normal(self.psi_unt[n], self.psi_unt_std), 0])
                    cnt.append(psi_unt)
                priv_unt.append(sum(cnt))
            else:
                priv_unt.append(0.)

            #Private tuned: 
            if self.priv_tun_bins[n] > 0:
                cnt = []
                for z in range(self.priv_tun_bins[n]):
                    psi_tun = np.max([0, np.dot(self.psi_tun[n, :], next_state)])
                    cnt.append(psi_tun)
                priv_tun.append(sum(cnt))

            else:
                priv_tun.append(0.)

        self.priv_tun = np.array(priv_tun)
        self.priv_unt = np.array(priv_unt)


        # Shar
        #Z are normally dist:
        if self.shar_tun_bins > 0:
            t_tun = []
            for z in range(self.shar_tun_bins):
                task_shar_tun = np.dot(self.U, np.dot(self.V , next_state))
                task_shar_tun[task_shar_tun<0] = 0.
                
                t_tun.append(task_shar_tun)
                
            self.shar_tun = np.squeeze(np.sum(np.hstack((t_tun)), axis=1))

        else: 
            self.shar_tun = np.zeros((self.n_neurons, ))

        self.unt_fact = np.random.normal(0, 1, (self.n_unt_factors, 1))
        if self.shar_unt_bins > 0:
            t_unt = []
            for z in range(self.shar_unt_bins):
                task_shar_unt = np.dot(self.W, self.unt_fact)
                task_shar_unt[task_shar_unt<0] = 0.
                t_unt.append(task_shar_unt)

            self.shar_unt = np.squeeze(np.sum(np.hstack((t_unt)), axis=1))
        else:
            self.shar_unt = np.zeros((self.n_neurons, ))


        #Now weight everything together:
        w = self.wt_sources
        counts = np.squeeze(np.array(w[0]*self.priv_unt + w[1]*self.priv_tun + w[2]*self.shar_unt + w[3]*self.shar_tun))
        counts = counts/self.lambda_spk_update

        if np.logical_or(mode=='ts', np.logical_and(mode is None, self.return_ts)):
            ts = []
            n_neurons = self.n_neurons
            for k, ind in enumerate(self.unit_inds):
                # separate spike counts into individual time-stamps
                n_spikes = int(counts[k])
                fake_time = (self.call_count + 0.5)* 1./60
                if n_spikes > 0:
                    #spike_data = [(fake_time, int(ind/4)+1, ind % 4) for m in range(n_spikes)] 
                    spike_data = [(fake_time, ind, 1) for m in range(n_spikes)] 
                    ts += (spike_data)

            ts = np.array(ts, dtype=ts_dtype)
            return ts
            
        elif np.logical_or(mode=='counts', np.logical_and(mode is None, self.return_ts is False)):
            return counts

    def mod_poisson(self, x, dt=0.1):
        x[x<0] = 0
        return poisson(x*dt)

    def y2_eq_r2_min_x2(self, x_arr, r2):
        y = []
        for x in x_arr:
            if np.random.random_sample() > 0.5:
                y.append(np.sqrt(r2 - x**2))
            else:
                y.append(-1*np.sqrt(r2 - x**2))
        return np.array(y)

class CursorVelCosEnc(GenericCosEnc):
    '''
    Cosine encoder tuned to the X-Z velocity of a cursor. Corresponds to the StateSpaceEndptVel2D state-space model
    '''
    def __init__(self, n_neurons=25, mod_depth=14./0.2, baselines=10, **kwargs):
        C = np.zeros([n_neurons, 7])
        C[:,-1] = baselines

        angles = np.linspace(0, 2 * np.pi, n_neurons)
        C[:,3] = mod_depth * np.cos(angles)
        C[:,5] = mod_depth * np.sin(angles)

        ssm = None
        super(CLDASimCosEnc, self).__init__(C, ssm=None, **kwargs)



#################################
##### Point-process encoder #####
#################################
class PointProcess(object):
    '''
    Simulate a single point process. Implemented by Suraj Gowda and Maryam Shanechi.
    '''
    def __init__(self, beta, dt, tau_samples=[], K=0, eps=1e-3):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        self.beta = beta.reshape(-1, 1)
        self.dt = dt
        self.tau_samples = tau_samples
        self.K = K
        self.eps = eps
        self.i = 0
        self.j = self.i + 1
        self.X = np.zeros([0, len(beta)])
        self._reset_res()
        self.tau = np.inf
        self.rate = np.nan

    def _exp_sample(self):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''              
        if len(self.tau_samples) > 0:
            self.tau = self.tau_samples.pop(0)
        else:
            u = np.random.rand()
            self.tau = np.log(1 - u);

    def _reset_res(self):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''              
        self.resold = 1000
        self.resnew = np.nan

    def _integrate_rate(self):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''              
        # integrate rate
        loglambda = np.dot(self.X[self.last_spike_ind:self.j+1, :], self.beta) #log of lambda delta
        # import pdb; pdb.set_trace()
        self.rate = np.ravel(np.exp(loglambda)/self.dt)

        if len(self.rate) > 2:
            self.resnew = self.tau + simps(self.rate, dx=self.dt, even='first')
        else:
            self.resnew = self.tau + trapz(self.rate, dx=self.dt)

    def _decide(self):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''              
        if (self.resold > 0) and (self.resnew > self.resold):
            return True
        else:
            #self.j = self.j + 1;
            self.resold = self.resnew;
            return False

    def _push(self, x_t):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        self.X = np.vstack([self.X, x_t])

    def __call__(self, x_t):
        '''
        Simulate whether the cell should fire at time t based on new stimulus x_t and previous stimuli (saved)
        
        Parameters
        ----------
        x_t : np.ndarray of size (N,)
            Current stimulus that the firing rate of the cell depends on.
            N should match the 
        
        Returns
        -------
        spiking_bin : bool
            True or false depending on whether the cell has fired after the present stimulus.
        '''              
        self._push(x_t)
        if np.abs(self.resold) < self.eps:
            spiking_bin = True
        else:
            self._integrate_rate()
            spiking_bin = self._decide()

        # Handle the spike
        if spiking_bin:
            self.last_spike_ind = self.j - 1
            self._reset_res()
            self._exp_sample()
            self._integrate_rate()
            self.resold = self.resnew;

        self.j += 1
        return spiking_bin

    def _init_sampling(self, x_t):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''              
        self._push(x_t) # initialize the observed extrinsic covariates
        self._reset_res()
        self._exp_sample()
        self.j = 1
        self.last_spike_ind = 0 # initialization

    def sim_batch(self, X, verbose=False):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''              
        framelength = X.shape[0]
        spikes = np.zeros(framelength);

        self._init_sampling(X[0,:])
    
        while self.j < framelength:
            #spiking_bin = self(X[self.j, :])
            spiking_bin = self.__call__(X[self.j, :])
            if self.j < framelength and spiking_bin:
                spikes[self.last_spike_ind] = 1;

        return spikes

class PointProcessEnsemble(object):
    '''
    Simulate an ensemble of point processes
    '''
    def __init__(self, beta, dt, init_state=None, tau_samples=None, eps=1e-3, 
                 hist_len=0, units=None):
        '''
        Constructor for PointProcessEnsemble
        
        Docstring    
        
        Parameters
        ----------
        beta : np.array of shape (n_units, n_covariates)
            Each row of the matrix specifies the relationship between a single point process in the ensemble and the common "stimuli"
        dt : float
             Sampling interval to integrate piont process likelihood over
        init_state : np.array, optional, default=[np.zeros(n_covariates-1), 1]
             Initial state of the common stimuli
        tau_samples : np.iterable, optional, default=None
             ARG_DESCR
        eps : DATA_TYPE, optional, default=0.001
             ARG_DESCR
        hist_len : DATA_TYPE, optional, default=0
             ARG_DESCR
        units : list of tuples, optional, default=None
             Identifiers for each element of the ensemble. One is automatically generated if none is provided
        
        Returns
        -------
        PointProcessEnsemble instance
        
        '''
        self.n_neurons, n_covariates = beta.shape
        if init_state == None:
            init_state = np.hstack([np.zeros(n_covariates - 1), 1])
        if tau_samples == None:
            tau_samples = [[]]*self.n_neurons
        point_process_units = []

        self.beta = beta

        for k in range(self.n_neurons):
            point_proc = PointProcess(beta[k,:], dt, tau_samples=tau_samples[k])
            point_proc._init_sampling(init_state)
            point_process_units.append(point_proc)

        self.point_process_units = point_process_units

        if units == None:
            self.units = np.vstack([(x, 1) for x in range(self.n_neurons)])
        else:
            self.units = units

    def get_units(self):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        return self.units

    def __call__(self, x_t):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        
        # x_t = np.hstack([x_t, 1])
        x_t = np.array(x_t).ravel()
        counts = np.array(map(lambda unit: unit(x_t), self.point_process_units)).astype(int)
        return counts

class CLDASimPointProcessEnsemble(PointProcessEnsemble):
    '''
    PointProcessEnsemble intended to be called at 60 Hz and return simulated
    spike timestamps at 180 Hz
    '''
    def __init__(self, *args, **kwargs):
        '''
        see PointProcessEnsemble.__init__
        '''
        super(CLDASimPointProcessEnsemble, self).__init__(*args, **kwargs)
        self.call_count = -1

    def __call__(self, x_t):
        '''
        Ensemble is called at 60 Hz but expects the timestamps to reflect spike
        bins determined at 180 Hz

        Parameters
        ----------
        x_t : np.ndarray

        
        Returns
        -------
        '''
        ts_data = []
        for k in range(3):
            counts = super(CLDASimPointProcessEnsemble, self).__call__(x_t)
            nonzero_units, = np.nonzero(counts)
            fake_time = self.call_count * 1./60 + (k + 0.5)*1./180
            for unit_ind in nonzero_units:
                ts = (fake_time, self.units[unit_ind, 0], self.units[unit_ind, 1], fake_time)
                ts_data.append(ts)

        self.call_count += 1
        return np.array(ts_data, dtype=ts_dtype_new)