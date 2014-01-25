import numpy as np
import time
from riglib.bmi import sim_neurons

ts_dtype_new = sim_neurons.ts_dtype_new

# object that gets the data that it needs (e.g., spikes, LFP, etc.) from the neural data source and 
# extracts features from it
class FeatureExtractor(object):
    '''Docstring.'''

    def __init__(self, task):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class BinnedSpikeCountsExtractor(FeatureExtractor):
    '''Docstring.'''
    feature_type = 'spike_counts'
    def __init__(self, source, n_subbins=1, units=[]):
        self.source = source
        self.n_subbins = n_subbins
        self.units = units
        self.last_get_spike_counts_time = 0

    def get_spike_ts(self, *args, **kwargs):
        return self.source.get()

    def get_bin_edges(self, ts):
        if len(ts) == 0:
            bin_edges = np.array([np.nan, np.nan])
        else:
            min_ind = np.argmin(ts['ts'])
            max_ind = np.argmax(ts['ts'])
            bin_edges = np.array([ts[min_ind]['ts'], ts[max_ind]['ts']])            

    def __call__(self, start_time, *args, **kwargs):
        ts = self.get_spike_ts(*args, **kwargs)
        if len(ts) == 0:
            counts = np.zeros([len(self.units), self.n_subbins])
        elif self.n_subbins > 1:
            subbin_edges = np.linspace(self.last_get_spike_counts_time, start_time, self.n_subbins+1)

            # Decrease the first subbin index to include any spikes that were
            # delayed in getting to the task layer due to threading issues
            # An acceptable delay is 1 sec or less. Realistically, most delays should be
            # on the millisecond order
            subbin_edges[0] -= 1
            subbin_inds = np.digitize(ts['arrival_ts'], subbin_edges)
            counts = np.vstack([bin_spikes(ts[subbin_inds == k], self.units) for k in range(1, self.n_subbins+1)]).T
        else:
            counts = bin_spikes(ts, self.units).reshape(-1, 1)

        counts = np.array(counts, dtype=np.uint32)
        bin_edges = self.get_bin_edges(ts)
        self.last_get_spike_counts_time = start_time
        return counts, bin_edges

class ReplaySpikeCountsExtractor(BinnedSpikeCountsExtractor):
    '''
    A "feature extractor" that replays spike counts from an HDF file
    '''    
    feature_type = 'spike_counts'
    def __init__(self, hdf_table, source='spike_counts', units=[]):
        self.idx = 0
        self.hdf_table = hdf_table
        self.source = source
        self.units = units
        self.n_subbins = hdf_table[0][source].shape[1]
        self.last_get_spike_counts_time = 0

    def get_spike_ts(self):
        # Get counts from HDF file
        counts = self.hdf_table[self.idx][self.source]
        n_subbins = counts.shape[1]

        # Convert counts to timestamps between (self.idx*1./60, (self.idx+1)*1./60)
        # NOTE: this code is mostly copied from riglib.bmi.sim_neurons.CLDASimPointProcessEnsemble
        ts_data = []
        for k in range(n_subbins):
            fake_time = (self.idx - 1) * 1./60 + (k + 0.5)*1./(60*n_subbins)
            nonzero_units, = np.nonzero(counts[:,k])
            for unit_ind in nonzero_units:
                n_spikes = counts[unit_ind, k]
                for m in range(n_spikes):
                    ts = (fake_time, self.units[unit_ind, 0], self.units[unit_ind, 1], fake_time)
                    ts_data.append(ts)

        return np.array(ts_data, dtype=ts_dtype_new)

    def get_bin_edges(self, ts):
        return self.hdf_table[self.idx]['bin_edges']

    def __call__(self, *args, **kwargs):
        output = super(ReplaySpikeCountsExtractor, self).__call__(*args, **kwargs)
        self.idx += 1 
        return output


def bin_spikes(ts, units, max_units_per_channel=13):
    '''
    Count up the number of BMI spikes in a list of spike timestamps.
    '''
    unit_inds = units[:,0]*max_units_per_channel + units[:,1]
    edges = np.sort(np.hstack([unit_inds - 0.5, unit_inds + 0.5]))
    spiking_unit_inds = ts['chan']*max_units_per_channel + ts['unit']
    counts, _ = np.histogram(spiking_unit_inds, edges)
    return counts[::2]

        



class LFPPowerExtractor(object):
    pass




