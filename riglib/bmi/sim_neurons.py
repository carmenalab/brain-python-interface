#!/usr/bin/python
"""
Collection of modules to simulate neural activity
"""
from __future__ import division
import os
import numpy as np

from scipy.io import loadmat

import numpy as np
from numpy.random import poisson, rand
from scipy.io import loadmat, savemat

from scipy.integrate import trapz, simps
import statsmodels.api as sm # TODO add back in for point process ensemble

ts_dtype = [('ts', float), ('chan', np.int32), ('unit', np.int32)]
class CosEnc():
    def __init__(self, n_neurons=25, mod_depth=14./0.2, baselines=10, 
        unit_inds=None, fname='', return_ts=False, DT=0.1):
        """Create neurons cosine-tuned to random directions."""

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
            self.angles = 2 * np.pi * np.random.rand(n_neurons)
            self.mod_depth = mod_depth
            self.pds = np.array([[np.cos(a), np.sin(a)] for a in self.angles])
            self.unit_inds = unit_inds
            self.save()

    def get_units(self):
        return np.array([ (int(ind/4)+1, ind % 4) for ind in self.unit_inds ])

    def __call__(self, user_input):
        """ Encode two-dimensional user input into firing rates.
        """
        if isinstance(self.baselines, np.ndarray):
            baselines = self.baselines.ravel()
        else:
            baselines = self.baselines
        rates = self.mod_depth * np.dot(self.pds, user_input) + baselines
        rates = np.array([max(r, 0) for r in rates])
        counts = poisson(rates * self.DT)

        if self.return_ts:
            ts = []
            n_neurons = self.n_neurons
            for k, ind in enumerate(self.unit_inds):
                # separate spike counts into individual time-stamps
                n_spikes = int(counts[k])
                if n_spikes > 0:
                    spike_data = [ (-1, int(ind/4)+1, ind % 4) for m in range(n_spikes)] 
                    ts += (spike_data)

            ts = np.array(ts, dtype=ts_dtype)
            return ts
        else:
            return counts

    def save(self):
        savemat(self.fname, {'n_neurons':self.n_neurons, 'baselines':self.baselines,
            'angles':self.angles, 'pds':self.pds, 'mod_depth':self.mod_depth,
            'unit_inds':self.unit_inds})


class PointProcess():
    def __init__(self, beta, dt, tau_samples=[], K=0, eps=1e-3):
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
        if len(self.tau_samples) > 0:
            self.tau = self.tau_samples.pop(0)
        else:
            print "drawing new samples"
            u = np.random.rand()
            self.tau = np.log(1 - u);

    def _reset_res(self):
        self.resold = 1000
        self.resnew = np.nan

    def _integrate_rate(self):
        # integrate rate
        loglambda = np.dot(self.X[self.last_spike_ind:self.j+1, :], self.beta) #log of lambda delta
        self.rate = np.ravel(np.exp(loglambda)/self.dt)

        if len(self.rate) > 2:
            self.resnew = self.tau + simps(self.rate, dx=self.dt, even='first')
        else:
            self.resnew = self.tau + trapz(self.rate, dx=self.dt)

    def _decide(self):
        if (self.resold > 0) and (self.resnew > self.resold):
            return True
        else:
            #self.j = self.j + 1;
            self.resold = self.resnew;
            return False

    def _push(self, x_t):
        self.X = np.vstack([self.X, x_t])

    def __call__(self, x_t):
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
        self._push(x_t) # initialize the observed extrinsic covariates
        self._reset_res()
        self._exp_sample()
        self.j = 1
        self.last_spike_ind = 0 # initialization

    def sim_batch(self, X, verbose=False):
        framelength = X.shape[0]
        spikes = np.zeros(framelength);

        self._init_sampling(X[0,:])
    
        while self.j < framelength:
            #spiking_bin = self(X[self.j, :])
            spiking_bin = self.__call__(X[self.j, :])
            if self.j < framelength and spiking_bin:
                spikes[self.last_spike_ind] = 1;

        return spikes


## class PointProcessEnsemble():
##     def __init__(self, beta, init_state, dt, tau_samples=None, eps=1e-3,
##                  hist_len=0, unit_inds=None):
##         '''
##         Initialize an ensemble of point-process neurons
##         '''
##         beta = np.vstack([beta[1:, :], beta[0,:]])
##         self.n_neurons = beta.shape[1]
##         if unit_inds == None:
##             unit_inds = np.arange(1, self.n_neurons+1)
##         self.unit_inds = unit_inds
## 
##         if tau_samples == None:
##             tau_samples = [[]] * self.n_neurons
## 
##         # TODO the individual point-processes need to be initialized with some 
##         # starting kinematic state (_init_sampling method)
##         self.units = []
##         for k in range(self.n_neurons):
##             point_proc = PointProcess(beta[:,k], dt, tau_samples[k])
##             point_proc._init_sampling(init_state)
##             self.units.append(point_proc)
##         #self.units = [ for k in range(self.n_neurons)]
##         self.include_offset = np.all(beta[0,:] == 1) # TODO baselines are logarithmed
##         self.beta = beta
## 
##     def __call__(self, user_input):
##         user_input = np.hstack([user_input, 1]) # TODO convention needs to go back to offset state last ...
##         return np.array([unit(user_input) for unit in self.units]).astype(np.float64).reshape(-1,1)


class PointProcessEnsemble():
    def __init__(self, beta, init_state, dt, tau_samples=None, eps=1e-3, 
                 hist_len=0, unit_inds=None):
        '''
        Initialize a point process ensemble
        '''
        self.n_neurons = beta.shape[1]
        units = []
        for k in range(self.n_neurons):
            point_proc = PointProcess(beta[:,k], dt, tau_samples=tau_samples[k])
            point_proc._init_sampling(init_state)
            units.append(point_proc)
        self.units = units

    def __call__(self, x_t):
        x_t = np.hstack([x_t, 1])
        return np.array(map(lambda unit: unit(x_t), self.units)).astype(int)
