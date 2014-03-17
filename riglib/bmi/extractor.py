import numpy as np
import time
from riglib.bmi import sim_neurons
from scipy.signal import butter, lfilter
import math

import nitime.algorithms as tsa


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
    '''
    Bins spikes using rectangular windows.
    '''

    feature_type = 'spike_counts'

    def __init__(self, source, n_subbins=1, units=[]):
        self.feature_dtype = ('spike_counts', 'u4', (len(units), n_subbins))

        self.source = source
        self.n_subbins = n_subbins
        self.units = units

        extractor_kwargs = dict()
        extractor_kwargs['n_subbins'] = self.n_subbins
        extractor_kwargs['units']     = self.units
        self.extractor_kwargs = extractor_kwargs

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

class SimBinnedSpikeCountsExtractor(BinnedSpikeCountsExtractor):
    '''Doctstring.'''
    
    def __init__(self, input_device, encoder, n_subbins, units):
        self.input_device = input_device
        self.encoder = encoder
        self.n_subbins = n_subbins
        self.units = units
        self.last_get_spike_counts_time = 0

    def get_spike_ts(self, cursor_pos, target_pos):
        ctrl    = self.input_device.get(target_pos, cursor_pos)
        ts_data = self.encoder(ctrl)
        return ts_data

    def __call__(self, start_time, cursor_pos, target_pos):
        return super(SimBinnedSpikeCountsExtractor, self).__call__(start_time, cursor_pos, target_pos)

def bin_spikes(ts, units, max_units_per_channel=13):
    '''
    Count up the number of BMI spikes in a list of spike timestamps.
    '''
    unit_inds = units[:,0]*max_units_per_channel + units[:,1]
    edges = np.sort(np.hstack([unit_inds - 0.5, unit_inds + 0.5]))
    spiking_unit_inds = ts['chan']*max_units_per_channel + ts['unit']
    counts, _ = np.histogram(spiking_unit_inds, edges)
    return counts[::2]

        

# bands should be a list of tuples representing ranges
#   e.g., bands = [(0, 10), (10, 20), (130, 140)] for 0-10, 10-20, and 130-140 Hz
# win_len specified in seconds

class LFPButterBPFPowerExtractor(object):
    '''
    Computes log power of the LFP in different frequency bands (for each 
    channel) in time-domain using Butterworth band-pass filters.
    '''

    feature_type = 'lfp_power'

    def __init__(self, source, channels=[], bands=[(10, 20), (20, 30)], win_len=0.2, filt_order=5, fs=1000):
        self.feature_dtype = ('lfp_power', 'u4', (len(channels)*len(bands), 1))

        self.source = source
        self.channels = channels
        self.bands = bands
        self.win_len = win_len
        self.filt_order = filt_order
        if source is not None:
            self.fs = source.source.update_freq
        else:
            self.fs = fs

        extractor_kwargs = dict()
        extractor_kwargs['channels']   = self.channels
        extractor_kwargs['bands']      = self.bands
        extractor_kwargs['win_len']    = self.win_len
        extractor_kwargs['filt_order'] = self.filt_order
        extractor_kwargs['fs']         = self.fs
        self.extractor_kwargs = extractor_kwargs

        self.n_pts = int(self.win_len * self.fs)
        self.filt_coeffs = dict()
        for band in bands:
            nyq = 0.5 * self.fs
            low = band[0] / nyq
            high = band[1] / nyq
            self.filt_coeffs[band] = butter(self.filt_order, [low, high], btype='band')  # returns (b, a)

        self.epsilon = 1e-9

        self.last_get_lfp_power_time = 0  # TODO -- is this variable necessary for LFP?

    def get_cont_samples(self, *args, **kwargs):
        return self.source.get(self.n_pts, self.channels)

    def extract_features(self, cont_samples):
        n_chan = len(self.channels)
        
        lfp_power = np.zeros((n_chan * len(self.bands), 1))
        for i, band in enumerate(self.bands):
            b, a = self.filt_coeffs[band]
            y = lfilter(b, a, cont_samples)
            lfp_power[i*n_chan:(i+1)*n_chan] = np.log((1. / self.n_pts) * np.sum(y**2, axis=1) + self.epsilon).reshape(-1, 1)

        return lfp_power

    def __call__(self, start_time, *args, **kwargs):
        cont_samples = self.get_cont_samples(*args, **kwargs)  # dims of channels x time
        lfp_power = self.extract_features(cont_samples)

        self.last_get_lfp_power_time = start_time
        
        # TODO -- what to return as equivalent of bin_edges?
        return lfp_power, None


class LFPMTMPowerExtractor(object):
    '''
    Computes log power of the LFP in different frequency bands (for each 
    channel) in freq-domain using the multi-taper method.
    '''

    feature_type = 'lfp_power'

    def __init__(self, source, channels=[], bands=[(10, 20), (20, 30)], win_len=0.2, NW=3, fs=1000):
        self.feature_dtype = ('lfp_power', 'u4', (len(channels)*len(bands), 1))

        self.source = source
        self.channels = channels
        self.bands = bands
        self.win_len = win_len
        self.NW = NW
        if source is not None:
            self.fs = source.source.update_freq
        else:
            self.fs = fs

        extractor_kwargs = dict()
        extractor_kwargs['channels'] = self.channels
        extractor_kwargs['bands']    = self.bands
        extractor_kwargs['win_len']  = self.win_len
        extractor_kwargs['NW']       = self.NW
        extractor_kwargs['fs']       = self.fs
        self.extractor_kwargs = extractor_kwargs

        self.n_pts = int(self.win_len * self.fs)
        self.nfft = 2**int(np.ceil(np.log2(self.n_pts)))  # nextpow2(self.n_pts)
        fft_freqs = np.arange(0., fs, float(fs)/self.nfft)[:self.nfft/2 + 1]
        self.fft_inds = dict()
        for band_idx, band in enumerate(bands):
            self.fft_inds[band_idx] = [freq_idx for freq_idx, freq in enumerate(fft_freqs) if band[0] <= freq < band[1]]

        self.epsilon = 1e-9

    def get_cont_samples(self, *args, **kwargs):
        return self.source.get(self.n_pts, self.channels)

    def extract_features(self, cont_samples):
        psd_est = tsa.multi_taper_psd(cont_samples, Fs=self.fs, NW=self.NW, jackknife=False, low_bias=True, NFFT=self.nfft)[1]
        
        # compute average power of each band of interest
        n_chan = len(self.channels)
        lfp_power = np.zeros((n_chan * len(self.bands), 1))
        for idx, band in enumerate(self.bands):
            lfp_power[idx*n_chan:(idx+1)*n_chan] = np.mean(np.log10(psd_est[:, self.fft_inds[idx]] + self.epsilon), axis=1).reshape(-1, 1)

        return lfp_power

    def __call__(self, start_time, *args, **kwargs):
        cont_samples = self.get_cont_samples(*args, **kwargs)  # dims of channels x time
        lfp_power = self.extract_features(cont_samples)

        # TODO -- what to return as equivalent of bin_edges?
        return lfp_power, None

