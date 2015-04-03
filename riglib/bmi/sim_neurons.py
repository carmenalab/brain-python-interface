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

class CosEnc(object):
    ''' Docstring '''
    def __init__(self, n_neurons=25, mod_depth=14./0.2, baselines=10, 
        unit_inds=None, fname='', return_ts=False, DT=0.1, angles=None):
        """
        Create neurons cosine-tuned to random directions.

        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------

        """

        self.return_ts = return_ts
        self.fname = fname
        self.DT = DT
        if unit_inds == None:
            unit_inds = np.arange(1, n_neurons+1)
        if fname and os.path.exists(fname):
            print "Reloading encoder from file: %s" % fname
            data = loadmat(fname)
            try:
                self.n_neurons = data['n_neurons'][0,0]
            except:
                self.n_neurons = int(data['n_neurons'])
            self.angles = data['angles'].reshape(-1)
            self.baselines = data['baselines']
            self.mod_depth = data['mod_depth']
            self.unit_inds = data['unit_inds'].ravel()
            if self.mod_depth.shape == (1,1):
                self.mod_depth = self.mod_depth[0,0]
            if self.baselines.shape == (1,1):
                self.baselines = self.baselines[0,0]
            self.pds = data['pds']
        else:
            self.n_neurons = n_neurons
            self.baselines = baselines
            self.angles = np.linspace(0, 2 * np.pi, n_neurons)
            self.mod_depth = mod_depth
            self.pds = np.array([[np.cos(a), np.sin(a)] for a in self.angles])
            self.unit_inds = unit_inds
            self.save()

    def get_units(self):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''

        return np.array([ (int(ind/4)+1, ind % 4) for ind in self.unit_inds ])

    def __call__(self, user_input):
        """ Encode two-dimensional user input into firing rates.
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------

        """
        if isinstance(self.baselines, np.ndarray):
            baselines = self.baselines.ravel()
        else:
            baselines = self.baselines
        rates = self.mod_depth * np.dot(self.pds, user_input) + baselines
        rates[rates < 0] = 0 # Floor firing rates at 0 Hz
        counts = poisson(rates * self.DT)

        if self.return_ts:
            ts = []
            n_neurons = self.n_neurons
            for k, ind in enumerate(self.unit_inds):
                # separate spike counts into individual time-stamps
                n_spikes = int(counts[k])
                fake_time = (self.call_count + 0.5)* 1./60
                if n_spikes > 0:
                    spike_data = [(fake_time, int(ind/4)+1, ind % 4) for m in range(n_spikes)] 
                    ts += (spike_data)

            ts = np.array(ts, dtype=ts_dtype)
            return ts
        else:
            return counts

    def save(self):
        '''
        Save the encoder parameters to a .mat file
        '''        
        savemat(self.fname, {'n_neurons':self.n_neurons, 'baselines':self.baselines,
            'angles':self.angles, 'pds':self.pds, 'mod_depth':self.mod_depth,
            'unit_inds':self.unit_inds})

class CLDASimCosEnc(CosEnc):
    ''' Docstring '''    
    def __init__(self, *args, **kwargs):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''        
        super(CLDASimCosEnc, self).__init__(*args, **kwargs)
        self.call_count = 0
        self.call_ds_rate = 6

    def __call__(self, user_input):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''        
        if self.call_count % self.call_ds_rate == 0: # TODO this assumes the neurons cannot change faster than 10 Hz
            ts_data = super(CLDASimCosEnc, self).__call__(user_input)
        elif self.return_ts:
            ts_data = np.array([])
        else:
            ts_data = np.zeros(self.n_neurons)
        self.call_count += 1
        return ts_data


class CosEncJoints(CosEnc):
    ''' Docstring '''
    def __init__(self, link_lengths, *args, **kwargs):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''        
        mod_depth = kwargs.pop('mod_depth', 14/0.2)
        kwargs['mod_depth'] = mod_depth/max(link_lengths)
        super(CosEncJoints, self).__init__(*args, **kwargs)

# TODO the class below is completely redundant!
class CLDASimCosEncJoints(CosEncJoints):
    ''' Docstring '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''              
        super(CLDASimCosEncJoints, self).__init__(*args, **kwargs)
        self.call_count = 0

    def __call__(self, user_input):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''              
        if self.call_count % 6 == 0: # TODO this assumes the neurons cannot change faster than 10 Hz
            ts_data = super(CLDASimCosEncJoints, self).__call__(user_input)
        elif self.return_ts:
            ts_data = np.array([])
        else:
            ts_data = np.zeros(self.n_neurons)
        self.call_count += 1
        return ts_data

class PointProcess():
    ''' Docstring '''
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
    ''' Docstring '''
    def __init__(self, beta, dt, init_state=None, tau_samples=None, eps=1e-3, 
                 hist_len=0, units=None):
        '''
        Initialize a point process ensemble
        
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        

        '''
        self.n_neurons, n_covariates = beta.shape
        if init_state == None:
            init_state = np.hstack([np.zeros(n_covariates - 1), 1])
        if tau_samples == None:
            tau_samples = [[]]*self.n_neurons
        point_process_units = []
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
        x_t = np.hstack([x_t, 1])
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
        

def load_ppf_encoder_2D_vel_tuning(fname, dt=0.005):
    '''
    Docstring    
        
    Parameters
    ----------
        
    Returns
    -------
    '''

    data = loadmat(fname)
    beta = data['beta']
    beta = np.vstack([beta[1:, :], beta[0,:]]).T
    n_neurons = beta.shape[0]

    init_state = np.array([0., 0, 1])

    try:
        tau_samples = [data['tau_samples'][0][k].ravel().tolist() for k in range(n_neurons)]
    except:
        tau_samples = []
    encoder = PointProcessEnsemble(beta, dt, init_state=init_state, tau_samples=tau_samples)
    return encoder

def load_ppf_encoder_2D_vel_tuning_clda_sim(fname, dt=0.005):
    '''
    Docstring    
        
    Parameters
    ----------
        
    Returns
    -------
    '''    
    data = loadmat(fname)
    beta = data['beta']
    beta = np.vstack([beta[1:, :], beta[0,:]]).T
    encoder = CLDASimPointProcessEnsemble(beta, dt)
    return encoder

