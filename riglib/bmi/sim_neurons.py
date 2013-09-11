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
import statsmodels.api as sm


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

