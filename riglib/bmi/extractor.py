'''
Classes for extracting "decodable features" from various types of neural signal sources. 
Examples include spike rate estimation, LFP power, and EMG amplitude.
'''
import numpy as np
from scipy.signal import butter, lfilter
import os
import nitime.algorithms as tsa
import aopy

class FeatureExtractor(object):
    '''
    Parent of all feature extractors, used only for interfacing/type-checking.
    Feature extractors are objects tha gets the data that it needs (e.g., spike timestamps, LFP voltages, etc.) 
    from the neural data source object and extracts features from it
    '''
    @classmethod
    def extract_from_file(cls, *args, **kwargs):
        raise NotImplementedError


class DummyExtractor(FeatureExtractor):
    '''
    An extractor which does nothing. Used for tasks which are only pretending to be BMI tasks, e.g., visual feedback tasks
    '''
    feature_type = 'obs'
    feature_dtype = [('obs', 'f8', (1,))]

    def __call__(self, *args, **kwargs):
        return dict(obs=np.array([[np.nan]]))

class DirectObsExtractor(FeatureExtractor):
    '''
    Doesn't 'extract' anything, just passes on the input
    '''
    feature_type = 'obs'

    def __init__(self, source):
        test = source.get()
        self.feature_dtype = [('obs', source.source.dtype, np.shape(test))]
        self.source = source

    def __call__(self, *args, **kwargs):
        return dict(obs=self.source.get())

class BinnedSpikeCountsExtractor(FeatureExtractor):
    '''
    Extracts spike counts from spike timestamps separated into rectangular window. 
    This extractor is (currently) the main type of feature extractor in intracortical BMIs
    '''
    feature_type = 'spike_counts'

    def __init__(self, source, n_subbins=1, units=[]):
        '''
        Constructor for BinnedSpikeCountsExtractor

        Parameters
        ----------
        source: DataSource instance
            Source must implement a '.get()' function which returns the appropriate data 
            (appropriateness will change depending on the source)
        n_subbins: int, optional, default=1
            Number of bins into which to divide the observed spike counts 
        units: np.ndarray of shape (N, 2), optional, default=[]
            Units which need spike binning. Each row of the array corresponds to (channel, unit). By default no units will be binned.

        Returns
        -------
        BinnedSpikeCountsExtractor instance
        '''
        self.feature_dtype = [('spike_counts', 'u4', (len(units), n_subbins)), ('bin_edges', 'f8', 2)]

        self.source = source
        self.n_subbins = n_subbins
        self.units = units

        extractor_kwargs = dict()
        extractor_kwargs['n_subbins'] = self.n_subbins
        extractor_kwargs['units']     = self.units
        self.extractor_kwargs = extractor_kwargs
        self.last_get_spike_counts_time = 0

    def set_n_subbins(self, n_subbins):
        '''
        Alter the # of subbins without changing the extractor kwargs of a decoder

        Parameters
        ----------
        n_subbins : int 
            Number of bins into which to divide the observed spike counts 

        Returns
        -------
        None
        '''
        self.n_subbins = n_subbins
        self.extractor_kwargs['n_subbins'] = n_subbins
        self.feature_dtype = [('spike_counts', 'u4', (len(self.units), n_subbins)), ('bin_edges', 'f8', 2)]

    def get_spike_ts(self, *args, **kwargs):
        '''
        Get the spike timestamps from the neural data source. This function has no type checking, 
        i.e., it is assumed that the Extractor object was created with the proper source

        Parameters
        ----------
        None are needed (args and kwargs are ignored)

        Returns
        -------
        numpy record array
            Spike timestamps in the format of riglib.plexon.Spikes.dtype
        '''
        return self.source.get()

    def get_bin_edges(self, ts):
        '''
        Determine the first and last spike timestamps to allow HDF files 
        created by the BMI to be semi-synchronized with the neural data file

        Parameters
        ----------
        ts : numpy record array
            Must have field 'ts' of spike timestamps in seconds

        Returns
        -------
        np.ndarray of shape (2,)
            The smallest and largest timestamps corresponding to the current feature; 
            useful for rough synchronization of the BMI event loop with the neural recording system.
        '''
        if len(ts) == 0:
            bin_edges = np.array([np.nan, np.nan])
        else:
            min_ind = np.argmin(ts['ts'])
            max_ind = np.argmax(ts['ts'])
            bin_edges = np.array([ts[min_ind]['ts'], ts[max_ind]['ts']])

    @classmethod
    def bin_spikes(cls, ts, units, max_units_per_channel=13):
        '''
        Count up the number of BMI spikes in a list of spike timestamps.

        Parameters
        ----------
        ts : numpy record array
            Must have field 'ts' of spike timestamps in seconds
        units : np.ndarray of shape (N, 2)
            Each row corresponds to the channel index (typically the electrode number) and 
            the unit index (an index to differentiate the possibly many units on the same electrode). These are 
            the units used in the BMI.
        max_units_per_channel : int, optional, default=13
            This int is used to map from a (channel, unit) index to a single 'unit_ind' 
            for faster binning of spike timestamps. Just set to a large number.

        Returns
        -------
        np.ndarray of shape (N, 1)
            Column vector of counts of spike events for each of the N units.
        '''
        unit_inds = units[:,0]*max_units_per_channel + units[:,1]
        edges = np.sort(np.hstack([unit_inds - 0.5, unit_inds + 0.5]))

        spiking_unit_inds = ts['chan']*max_units_per_channel + ts['unit']
        counts, _ = np.histogram(spiking_unit_inds, edges)
        return counts[::2]

    def __call__(self, start_time, *args, **kwargs):
        '''
        Main function to retreive new spike data and bin the counts

        Parameters
        ----------
        start_time : float 
            Absolute time from the task event loop. This is used only to subdivide 
            the spike timestamps into multiple bins, if desired (if the 'n_subbins' attribute is > 1)
        *args, **kwargs : optional positional/keyword arguments
            These are passed to the source, or ignored (not needed for this extractor).

        Returns
        -------
        dict
            Extracted features to be saved in the task. 
        '''
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
            counts = np.vstack([self.bin_spikes(ts[subbin_inds == k], self.units) for k in range(1, self.n_subbins+1)]).T
        else:
            counts = self.bin_spikes(ts, self.units).reshape(-1, 1)

        counts = np.array(counts, dtype=np.uint32)
        bin_edges = self.get_bin_edges(ts)
        self.last_get_spike_counts_time = start_time

        return dict(spike_counts=counts, bin_edges=bin_edges)

    @classmethod
    def extract_from_file(cls, files, neurows, binlen, units, extractor_kwargs, strobe_rate=60.0):
        '''
        Compute binned spike count features

        Parameters
        ----------
        files : dict
            Data files used to train the decoder. Should contain exactly one type of neural data file (e.g., Plexon, Blackrock, TDT)
        neurows: np.ndarray of shape (T,)
            Timestamps in the plexon time reference corresponding to bin boundaries
        binlen: float
            Length of time over which to sum spikes from the specified cells
        units: np.ndarray of shape (N, 2)
            List of units that the decoder will be trained on. The first column specifies the electrode number and the second specifies the unit on the electrode
        extractor_kwargs: dict 
            Any additional parameters to be passed to the feature extractor. This function is agnostic to the actual extractor utilized
        strobe_rate: 60.0
            The rate at which the task sends the sync pulse to the plx file

        Returns
        -------
        spike_counts : np.ndarray of shape (N, T)
            Spike counts binned over the length of the datafile.
        units : np.ndarray of shape (N, 2)
            Each row corresponds to the channel index (typically the electrode number) and 
            the unit index (an index to differentiate the possibly many units on the same electrode). These are 
            the units used in the BMI.
        extractor_kwargs : dict
            Parameters used to instantiate the feature extractor, to be stored 
            along with the trained decoder so that the exact same feature extractor can be re-created at runtime.
        '''
        if 'plexon' in files:
            from plexon import plexfile
            plx = plexfile.openFile(files['plexon'].encode('utf-8'))
            # interpolate between the rows to 180 Hz
            if binlen < 1./strobe_rate:
                interp_rows = []
                neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
                for r1, r2 in zip(neurows[:-1], neurows[1:]):
                    interp_rows += list(np.linspace(r1, r2, 4)[1:])
                interp_rows = np.array(interp_rows)
            else:
                step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
                interp_rows = neurows[::step]
            print(('step: ', step))
            from plexon import psth
            spike_bin_fn = psth.SpikeBin(units, binlen)
            spike_counts = np.array(list(plx.spikes.bin(interp_rows, spike_bin_fn)))


            # discard units that never fired at all
            discard_zero_units = extractor_kwargs.pop('discard_zero_units', True)
            if discard_zero_units:
                unit_inds, = np.nonzero(np.sum(spike_counts, axis=0))
                units = units[unit_inds,:]
                spike_counts = spike_counts[:, unit_inds]
                extractor_kwargs['units'] = units

            return spike_counts, units, extractor_kwargs

        elif 'blackrock' in files:
            nev_fname = [name for name in files['blackrock'] if '.nev' in name][0]  # only one of them
            nev_hdf_fname = [name for name in files['blackrock'] if '.nev' in name and name[-4:]=='.hdf']
            nsx_fnames = [name for name in files['blackrock'] if '.ns' in name]            
            # interpolate between the rows to 180 Hz
            if binlen < 1./strobe_rate:
                interp_rows = []
                neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
                for r1, r2 in zip(neurows[:-1], neurows[1:]):
                    interp_rows += list(np.linspace(r1, r2, 4)[1:])
                interp_rows = np.array(interp_rows)
            else:
                step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
                interp_rows = neurows[::step]

            
            
            if len(nev_hdf_fname) == 0:
                nev_hdf_fname = nev_fname + '.hdf'
                
                if not os.path.isfile(nev_hdf_fname):
                    # convert .nev file to hdf file using Blackrock's n2h5 utility
                    subprocess.call(['n2h5', nev_fname, nev_hdf_fname])
            else:
                nev_hdf_fname = nev_hdf_fname[0]

            try:
                nev_hdf = h5py.File(nev_hdf_fname, 'r')
                open_method = 1
            except:
                import tables
                nev_hdf = tables.openFile(nev_hdf_fname)
                open_method = 2
                #print 'open method 2'

            n_bins = len(interp_rows)
            n_units = units.shape[0]
            spike_counts = np.zeros((n_bins, n_units))

            for i in range(n_units):
                chan = units[i, 0]

                # 1-based numbering (comes from web interface)
                unit = units[i, 1]  

                chan_str = str(chan).zfill(5)
                path = 'channel/channel%s/spike_set' % chan_str

                if open_method == 1:    
                    ts = nev_hdf.get(path).value['TimeStamp']
                    # the units corresponding to each timestamp in ts
                    # 0-based numbering (comes from .nev file), so add 1
                    units_ts = nev_hdf.get(path).value['Unit']
                
                elif open_method == 2:
                    try:
                        grp = nev_hdf.getNode('/'+path)
                        ts = grp[:]['TimeStamp']
                        units_ts = grp[:]['Unit']
                    except:
                        print(('no spikes recorded on channel: ', chan_str, ': adding zeros'))
                        ts = []
                        units_ts = []


                # get the ts for this unit, in units of secs
                fs = 30000.
                ts = [t/fs for idx, (t, u_t) in enumerate(zip(ts, units_ts)) if u_t == unit]

                # insert value interp_rows[0]-step to beginning of interp_rows array
                interp_rows_ = np.insert(interp_rows, 0, interp_rows[0]-step)

                # use ts to fill in the spike_counts that corresponds to unit i
                spike_counts[:, i] = np.histogram(ts, interp_rows_)[0]


            # discard units that never fired at all
            if 'keep_zero_units' in extractor_kwargs:
                print('keeping zero firing units')
            else:
                unit_inds, = np.nonzero(np.sum(spike_counts, axis=0))
                units = units[unit_inds,:]
                spike_counts = spike_counts[:, unit_inds]
            
            extractor_kwargs['units'] = units

            return spike_counts, units, extractor_kwargs  
                 
        elif 'tdt' in files:
            raise NotImplementedError     

        elif 'ecube' in files:
            raise NotImplementedError

# bands should be a list of tuples representing ranges
#   e.g., bands = [(0, 10), (10, 20), (130, 140)] for 0-10, 10-20, and 130-140 Hz
start = 0
end   = 150
step  = 10
default_bands = []
for freq in range(start, end, step):
    default_bands.append((freq, freq+step))

class LFPMTMPowerExtractor(object):
    '''
    Computes log power of the LFP in different frequency bands (for each 
    channel) in freq-domain using the multi-taper method.
    '''

    feature_type = 'lfp_power'

    def __init__(self, source, channels=[], bands=default_bands, win_len=0.2, NW=3, fs=1000, **kwargs):
        '''
        Constructor for LFPMTMPowerExtractor, which extracts LFP power using the multi-taper method

        Parameters
        ----------
        source : riglib.source.Source object
            Object which yields new data when its 'get' method is called
        channels : list 
            LFP electrode indices to use for feature extraction
        bands : list of tuples
            Each tuple defines a frequency band of interest as (start frequency, end frequency) 

        Returns
        -------
        LFPMTMPowerExtractor instance
        '''
        #self.feature_dtype = ('lfp_power', 'f8', (len(channels)*len(bands), 1))

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
        self.npk = aopy.precondition.convert_taper_parameters(win_len, NW/win_len)
   
        extractor_kwargs['no_log']  = 'no_log' in kwargs and kwargs['no_log']==True #remove log calculation
        extractor_kwargs['no_mean'] = 'no_mean' in kwargs and kwargs['no_mean']==True #r
        self.extractor_kwargs = extractor_kwargs

        self.n_pts = int(self.win_len * self.fs)
        self.nfft = 2**int(np.ceil(np.log2(self.n_pts)))  # nextpow2(self.n_pts)
        fft_freqs = np.arange(0., fs, float(fs)/self.nfft)[:int(self.nfft/2) + 1]
        self.fft_inds = dict()
        for band_idx, band in enumerate(bands):
            self.fft_inds[band_idx] = [freq_idx for freq_idx, freq in enumerate(fft_freqs) if band[0] <= freq < band[1]]

        extractor_kwargs['fft_inds']       = self.fft_inds
        extractor_kwargs['fft_freqs']      = fft_freqs
        
        self.epsilon = 1e-9

        if extractor_kwargs['no_mean']: #Used in lfp 1D control task
            self.feature_dtype = ('lfp_power', 'f8', (len(channels)*len(fft_freqs), 1))
        else:
            self.feature_dtype = ('lfp_power', 'f8', (len(channels)*len(bands), 1))

    def get_cont_samples(self, *args, **kwargs):
        '''
        Retreives the last n_pts number of samples for each LPF channel from the neural data 'source'

        Parameters
        ----------
        *args, **kwargs : optional arguments
            Ignored for this extractor (not necessary)

        Returns
        -------
        np.ndarray of shape ???
        '''
        return self.source.get(self.n_pts, self.channels)

    # def extract_features(self, cont_samples):
    #     '''
    #     Extract spectral features from a block of time series samples

    #     Parameters
    #     ----------
    #     cont_samples : np.ndarray of shape (n_channels, n_samples)
    #         Raw voltage time series (one per channel) from which to extract spectral features 

    #     Returns
    #     -------
    #     lfp_power : np.ndarray of shape (n_channels * n_features, 1)
    #         Multi-band power estimates for each channel, for each band specified when the feature extractor was instantiated.
    #     '''
    #     psd_est = tsa.multi_taper_psd(cont_samples, Fs=self.fs, NW=self.NW, jackknife=False, low_bias=True, NFFT=self.nfft)[1]
        
    #     if ('no_mean' in self.extractor_kwargs) and (self.extractor_kwargs['no_mean'] is True):
    #         return psd_est.reshape(psd_est.shape[0]*psd_est.shape[1], 1)

    #     else:
    #         # compute average power of each band of interest
    #         n_chan = len(self.channels)
    #         lfp_power = np.zeros((n_chan * len(self.bands), 1))
    #         for idx, band in enumerate(self.bands):
    #             if self.extractor_kwargs['no_log']:
    #                 lfp_power[idx*n_chan : (idx+1)*n_chan, 0] = np.mean(psd_est[:, self.fft_inds[idx]], axis=1)
    #             else:
    #                 lfp_power[idx*n_chan : (idx+1)*n_chan, 0] = np.mean(np.log10(psd_est[:, self.fft_inds[idx]] + self.epsilon), axis=1)
            
    #         return lfp_power
        
    def extract_features(self, cont_samples):
        '''
        Extract spectral features from a block of time series samples

        Parameters
        ----------
        cont_samples : np.ndarray of shape (n_channels, n_samples)
            Raw voltage time series (one per channel) from which to extract spectral features 

        Returns
        -------
        lfp_power : np.ndarray of shape (n_channels * n_features, 1)
            Multi-band power estimates for each channel, for each band specified when the feature extractor was instantiated.
        '''
        assert int(self.win_len * self.fs) == cont_samples.shape[1]

        freqs, time, psd_est = aopy.analysis.calc_mt_tfr(cont_samples.T, *self.npk, self.fs, 1)

        if ('no_mean' in self.extractor_kwargs) and (self.extractor_kwargs['no_mean'] is True):
            return psd_est.reshape(psd_est.shape[0]*psd_est.shape[1], 1)

        else:
            # compute average power of each band of interest
            lfp_power = aopy.analysis.get_tfr_feats(freqs, psd_est, self.bands, not self.extractor_kwargs['no_log'], epsilon=self.epsilon)
            if lfp_power.ndim > 1:
                return lfp_power.reshape(lfp_power.shape[0]*lfp_power.shape[1], 1)
            else:
                return lfp_power.reshape(lfp_power.size, 1)

    def __call__(self, start_time, *args, **kwargs):
        '''
        Parameters
        ----------
        start_time : float 
            Absolute time from the task event loop. This is unused by LFP extractors in their current implementation
            and only passed in to ensure that function signatures are the same across extractors.
        *args, **kwargs : optional positional/keyword arguments
            These are passed to the source, or ignored (not needed for this extractor).

        Returns
        -------
        dict
            Extracted features to be saved in the task.         
        '''
        cont_samples = self.get_cont_samples(*args, **kwargs)  # dims of channels x time
        lfp_power = self.extract_features(cont_samples)

        return dict(lfp_power=lfp_power)

    @classmethod
    def extract_from_file(cls, files, neurows, binlen, units, extractor_kwargs, strobe_rate=60.0):
        '''
        Compute binned spike count features

        Parameters
        ----------
        files : dict
            Data files used to train the decoder. Should contain exactly one type of neural data file (e.g., Plexon, Blackrock, TDT)
        neurows: np.ndarray of shape (T,)
            Timestamps in the plexon time reference corresponding to bin boundaries
        binlen: float
            Length of time over which to sum spikes from the specified cells
        units: np.ndarray of shape (N, 2)
            List of units that the decoder will be trained on. The first column specifies the electrode number and the second specifies the unit on the electrode
        extractor_kwargs: dict 
            Any additional parameters to be passed to the feature extractor. This function is agnostic to the actual extractor utilized
        strobe_rate: 60.0
            The rate at which the task sends the sync pulse to the plx file

        Returns
        -------
        spike_counts : np.ndarray of shape (N, T)
            Spike counts binned over the length of the datafile.
        units : 
            Not used by this type of extractor, just passed back from the input argument to make the outputs consistent with spike count extractors
        extractor_kwargs : dict
            Parameters used to instantiate the feature extractor, to be stored 
            along with the trained decoder so that the exact same feature extractor can be re-created at runtime.
        '''
    
        # interpolate between the rows to 180 Hz
        if binlen < 1./strobe_rate:
            interp_rows = []
            neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
            for r1, r2 in zip(neurows[:-1], neurows[1:]):
                interp_rows += list(np.linspace(r1, r2, 4)[1:])
            interp_rows = np.array(interp_rows)
        else:
            step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
            interp_rows = neurows[::step]
        
        # create extractor object
        f_extractor = LFPMTMPowerExtractor(None, **extractor_kwargs)
        extractor_kwargs = f_extractor.extractor_kwargs

        win_len  = f_extractor.win_len
        bands    = f_extractor.bands
        channels = f_extractor.channels
        fs       = f_extractor.fs
        print(('bands:', bands))
        print(('win_len:', win_len))

        n_itrs = len(interp_rows)
        n_chan = len(channels)
        lfp_power = np.zeros((n_itrs, n_chan * len(bands)))

        # get the samples
        if 'plexon' in files:
            from plexon import plexfile
            plx = plexfile.openFile(files['plexon'].encode('utf-8'))
            lfp = plx.lfp[:].data[:, channels-1]

        elif 'blackrock' in files:
            raise NotImplementedError

        elif 'ecube' in files:
            from riglib.ecube.file import load_lfp
            data = load_lfp(str(files['ecube']))
            lfp = data[:, [c-1 for c in channels]]
        
        # for i, t in enumerate(interp_rows):
        #     cont_samples = plx.lfp[t-win_len:t].data[:, channels-1]
        #     lfp_power[i, :] = f_extractor.extract_features(cont_samples.T).T
        n_pts = int(win_len * fs)
        for i, t in enumerate(interp_rows):
            # try:
            sample_num = int(t * fs)
            cont_samples = lfp[max(0,sample_num-n_pts):min(lfp.shape[0], sample_num), :]
            if cont_samples.shape[0] < n_pts:
                # maybe don't count these ones?
                continue
            lfp_power[i, :] = f_extractor.extract_features(cont_samples.T).T
            # except:
            #     print("Error with LFP decoder training")
            #     print((i, t))


        # TODO -- discard any channel(s) for which the log power in any frequency 
        #   bands was ever equal to -inf (i.e., power was equal to 0)
        # or, perhaps just add a small epsilon inside the log to avoid this
        # then, remember to do this:  extractor_kwargs['channels'] = channels
        #   and reset the units variable

        return lfp_power, units, extractor_kwargs


#########################################################
##### Reconstruction extractors, used in test cases #####
#########################################################
class ReplaySpikeCountsExtractor(BinnedSpikeCountsExtractor):
    '''
    A "feature extractor" that replays spike counts from an HDF file
    '''
    feature_type = 'spike_counts'
    def __init__(self, hdf_table, source='spike_counts', cycle_rate=60.0, units=[]):
        '''
        Parameters
        ----------
        hdf_table : HDF table
            Data table to replay. Usually the 'task' table. 
        source : string, optional, default=spike_counts
            Column of the HDF table to replay
        cycle_rate : float, optional, default=60.0
            Rate at which the task FSM "cycles", i.e., the rate at which the task will ask for new observations
        units : iterable, optional, default=[]
            Names (channel, unit) of the units. If none specified, some fake names are created
    
        Returns
        -------
        ReplaySpikeCountsExtractor instance
        '''
        self.idx = 0
        self.hdf_table = hdf_table
        self.source = source
        self.units = units
        self.n_subbins = hdf_table[0][source].shape[1]
        self.last_get_spike_counts_time = 0
        self.cycle_rate = cycle_rate

        n_units = hdf_table[0]['spike_counts'].shape[0]
        self.feature_dtype = [('spike_counts', 'u4', (n_units, self.n_subbins)), 
                              ('bin_edges', 'f8', 2)]

    def get_spike_ts(self):
        '''
        Make up fake timestamps to go with the spike counts extracted from the HDF file
        '''
        from . import sim_neurons

        # Get counts from HDF file
        counts = self.hdf_table[self.idx][self.source]
        n_subbins = counts.shape[1]

        # Convert counts to timestamps between (self.idx*1./cycle_rate, (self.idx+1)*1./cycle_rate)
        # NOTE: this code is mostly copied from riglib.bmi.sim_neurons.CLDASimPointProcessEnsemble
        ts_data = []
        cycle_rate = self.cycle_rate
        for k in range(n_subbins):
            fake_time = (self.idx - 1) * 1./cycle_rate + (k + 0.5)*1./cycle_rate*1./n_subbins
            nonzero_units, = np.nonzero(counts[:,k])
            for unit_ind in nonzero_units:
                n_spikes = counts[unit_ind, k]
                for m in range(n_spikes):
                    ts = (fake_time, self.units[unit_ind, 0], self.units[unit_ind, 1], fake_time)
                    ts_data.append(ts)

        ts_dtype_new = sim_neurons.ts_dtype_new
        return np.array(ts_data, dtype=ts_dtype_new)

    def get_bin_edges(self, ts):
        '''
        Get the first and last timestamp of spikes in the current "bin" as saved in the HDF file
        '''
        return self.hdf_table[self.idx]['bin_edges']

    def __call__(self, *args, **kwargs):
        '''
        See BinnedSpikeCountsExtractor.__call__ for documentation
        '''
        output = super(ReplaySpikeCountsExtractor, self).__call__(*args, **kwargs)
        if not np.array_equal(output['spike_counts'], self.hdf_table[self.idx][self.source]):
            print(("spike binning error: ", self.idx))
        self.idx += 1 
        return output

class ReplayLFPPowerExtractor(LFPMTMPowerExtractor):
    '''
    A "feature extractor" that replays LFP power estimates from an HDF file
    '''

    def __init__(self, hdf_table, source='lfp_power'):
        '''    
        Constructor for ReplayLFPPowerExtractor

        Parameters
        ----------
        hdf_table : HDF table
            Data table to replay. Usually the 'task' table. 
        source : string, optional, default=spike_counts
            Column of the HDF table to replay
    
        Returns
        -------
        ReplayLFPPowerExtractor instance
        '''
        self.idx = 0
        self.hdf_table = hdf_table
        self.source = source
        self.n_subbins = hdf_table[0][source].shape[1]
        self.last_get_spike_counts_time = 0

        n_units = hdf_table[0][source].shape[0]
        self.feature_dtype = [('lfp_power', 'f8', (n_units, self.n_subbins)), ]

    def __call__(self, *args, **kwargs):
        '''    
        See LFPMTMPowerExtractor.__call__ for documentation
        '''
        output = self.hdf_table[self.idx][self.source]
        self.idx += 1 
        return dict(lfp_power=output)

#################################
##### Simulation extractors #####
#################################
class SimBinnedSpikeCountsExtractor(BinnedSpikeCountsExtractor):
    '''
    Spike count features are generated by a population of synthetic neurons
    '''
    feature_type = 'spike_counts'

    def __init__(self, input_device, encoder, n_subbins, units, task=None):
        '''
        Constructor for SimBinnedSpikeCountsExtractor

        Parameters
        ----------
        input_device: object with a "calc_next_state" method
            Generate the "intended" next state, e.g., by feedback control policy
        encoder: callable with 1 argument
            Maps the "control" input into the spike timestamps of a set of neurons
        n_subbins:
            Number of subbins to divide the spike counts into, e.g. 3 are necessary for the PPF
        units: np.ndarray of shape (N, 2)
            Each row of the array corresponds to (channel, unit)

        Returns
        -------
        SimBinnedSpikeCountsExtractor instance
        '''
        self.input_device = input_device
        self.encoder = encoder
        self.n_subbins = n_subbins
        self.units = units
        self.last_get_spike_counts_time = 0
        self.feature_dtype = [('spike_counts', 'f8', (len(units), n_subbins)), ('bin_edges', 'f8', 2),
            ('ctrl_input', 'f8', self.encoder.C.shape[1])]
        self.task = task
        self.sim_ctrl = np.zeros((self.encoder.C.shape[1]))
        
    def get_spike_ts(self):
        '''
        see BinnedSpikeCountsExtractor.get_spike_ts for docs
        '''
        current_state = self.task.get_current_state()
        target_state = self.task.get_target_BMI_state()
        ctrl = self.input_device.calc_next_state(current_state, target_state)
        #print current_state.T, target_state.T, ctrl.T
        self.sim_ctrl = ctrl
        ts_data = self.encoder(ctrl)
        return ts_data

class SimPowerExtractor(LFPMTMPowerExtractor):
    '''
    This extractor pretends to generate MTMPower estimates but really it just  
    fetches the powers from a given encoder at the appropriate freq bands
    '''
    def __init__(self, input_device, encoder, channels=[], bands=default_bands, task=None, **kwargs):
        '''
        Constructor for SimPowerExtractor

        Parameters
        ----------
        input_device: object with a "calc_next_state" method
            Generate the "intended" next state, e.g., by feedback control policy
        encoder: callable with 1 argument
            Maps the "control" input into the spike timestamps of a set of neurons
        channels : list 
            LFP electrode indices to use for feature extraction
        bands : list of tuples
            Each tuple defines a frequency band of interest as (start frequency, end frequency) 

        Returns
        -------
        SimPowerExtractor instance
        '''
        self.input_device = input_device
        self.encoder = encoder
        self.channels = channels
        self.bands = bands
        self.feature_dtype = ('lfp_power', 'f8', (len(channels)*len(bands), 1))
        self.task = task
        extractor_kwargs = dict()
        extractor_kwargs['channels']   = channels
        extractor_kwargs['bands']      = bands
        self.extractor_kwargs = extractor_kwargs
        
    def get_cont_samples(self, *args, **kwargs):
        '''
        see LFPMTMPowerExtractor.get_cont_samples for docs
        '''
        current_state = self.task.get_current_state()
        target_state = self.task.get_target_BMI_state()
        ctrl = self.input_device.calc_next_state(current_state, target_state)
        cont_data = self.encoder(ctrl)
        return cont_data

    def __call__(self, start_time, *args, **kwargs):
        '''
        Parameters
        ----------
        start_time : float 
            Absolute time from the task event loop. This is unused by LFP extractors in their current implementation
            and only passed in to ensure that function signatures are the same across extractors.
        *args, **kwargs : optional positional/keyword arguments
            These are passed to the source, or ignored (not needed for this extractor).

        Returns
        -------
        dict
            Extracted features to be saved in the task.         
        '''
        lfp_power = self.get_cont_samples(*args, **kwargs)  # dims of channels x bands
        return dict(lfp_power=lfp_power)

class SimDirectObsExtractor(SimBinnedSpikeCountsExtractor):
    '''
    This extractor just passes back the observation vector generated by the encoder
    '''
    def __call__(self, start_time, *args, **kwargs):
        y_t = self.get_spike_ts(*args, **kwargs)
        return dict(spike_counts=np.reshape(y_t, (len(self.units),self.n_subbins)))


#############################################
##### Feature extractors in development #####
#############################################
class LFPButterBPFPowerExtractor(object):
    '''
    Computes log power of the LFP in different frequency bands (for each 
    channel) in time-domain using Butterworth band-pass filters.
    '''

    feature_type = 'lfp_power'

    def __init__(self, source, channels=[], bands=default_bands, win_len=0.2, filt_order=5, fs=1000):
        self.feature_dtype = ('lfp_power', 'u4', (len(channels)*len(bands), 1))

        self.source = source
        self.channels = channels
        self.bands = bands
        self.win_len = win_len  # secs
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
        
        return dict(lfp_power=lfp_power)

    @classmethod
    def extract_from_file(cls, files, neurows, binlen, units, extractor_kwargs, strobe_rate=60.0):
        '''Compute lfp power features from a blackrock data file.'''

        nsx_fnames = [name for name in files['blackrock'] if '.ns' in name]

        # interpolate between the rows to 180 Hz
        if binlen < 1./strobe_rate:
            interp_rows = []
            neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
            for r1, r2 in zip(neurows[:-1], neurows[1:]):
                interp_rows += list(np.linspace(r1, r2, 4)[1:])
            interp_rows = np.array(interp_rows)
        else:
            step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
            interp_rows = neurows[::step]

        # TODO -- for now, use .ns3 or .ns2 file (2 kS/s)
        nsx_fname = None
        for fname in nsx_fnames:
            if '.ns3' in fname:
                nsx_fname = fname
                fs_ = 2000
        if nsx_fname is None:
            for fname in nsx_fnames:
                if '.ns2' in fname:
                    nsx_fname = fname
                    fs_ = 1000
        
        if nsx_fname is None:
            raise Exception('Need an nsx file --> .ns2 or .ns3 is acceptable. Higher nsx files yield memory errors')
        extractor_kwargs['fs'] = fs_

        # default order of 5 seems to cause problems when fs > 1000
        extractor_kwargs['filt_order'] = 3

        if nsx_fname[-4:] == '.hdf':
            nsx_hdf_fname = nsx_fname
        else:
            nsx_hdf_fname = nsx_fname + '.hdf'
            if not os.path.isfile(nsx_hdf_fname):
                # convert .nsx file to hdf file using Blackrock's n2h5 utility
                from db.tracker import models
                models.parse_blackrock_file(None, [nsx_fname], )

        import h5py
        nsx_hdf = h5py.File(nsx_hdf_fname, 'r')

        # create extractor object
        f_extractor = LFPButterBPFPowerExtractor(None, **extractor_kwargs)
        extractor_kwargs = f_extractor.extractor_kwargs

        win_len  = f_extractor.win_len
        bands    = f_extractor.bands
        channels = f_extractor.channels
        fs       = f_extractor.fs

        n_itrs = len(interp_rows)
        n_chan = len(channels)
        lfp_power = np.zeros((n_itrs, n_chan * len(bands)))
        n_pts = int(win_len * fs)

        # for i, t in enumerate(interp_rows):
        #     sample_num = int(t * fs)
        #     # cont_samples = np.zeros((n_chan, n_pts))

        #     # for j, chan in enumerate(channels):
        #     #     chan_str = str(chan).zfill(5)
        #     #     path = 'channel/channel%s/continuous_set' % chan_str
        #     #     cont_samples[j, :] = nsx_hdf.get(path).value[sample_num-n_pts:sample_num]
        #     cont_samples = abs(np.random.randn(n_chan, n_pts))

        #     feats = f_extractor.extract_features(cont_samples).T
        #     print feats
        #     lfp_power[i, :] = f_extractor.extract_features(cont_samples).T
        
        print(('*' * 40))
        print('WARNING: replacing LFP values from .ns3 file with random values!!')
        print(('*' * 40))

        lfp_power = abs(np.random.randn(n_itrs, n_chan * len(bands)))

        # TODO -- discard any channel(s) for which the log power in any frequency 
        #   bands was ever equal to -inf (i.e., power was equal to 0)
        # or, perhaps just add a small epsilon inside the log to avoid this
        # then, remember to do this:  extractor_kwargs['channels'] = channels
        #   and reset the units variable
        
        return lfp_power, units, extractor_kwargs


class AIMTMPowerExtractor(LFPMTMPowerExtractor):
    ''' Multitaper extractor for Plexon analog input channels'''

    feature_type = 'ai_power'

    def __init__(self, source, channels=[], bands=default_bands, win_len=0.2, NW=3, fs=1000, **kwargs):
        #self.feature_dtype = ('lfp_power', 'f8', (len(channels)*len(bands), 1))

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

   
        extractor_kwargs['no_log']  = 'no_log' in kwargs and kwargs['no_log']==True #remove log calculation
        extractor_kwargs['no_mean'] = 'no_mean' in kwargs and kwargs['no_mean']==True #r
        self.extractor_kwargs = extractor_kwargs

        self.n_pts = int(self.win_len * self.fs)
        self.nfft = 2**int(np.ceil(np.log2(self.n_pts)))  # nextpow2(self.n_pts)
        fft_freqs = np.arange(0., fs, float(fs)/self.nfft)[:self.nfft/2 + 1]
        self.fft_inds = dict()
        for band_idx, band in enumerate(bands):
            self.fft_inds[band_idx] = [freq_idx for freq_idx, freq in enumerate(fft_freqs) if band[0] <= freq < band[1]]

        extractor_kwargs['fft_inds']       = self.fft_inds
        extractor_kwargs['fft_freqs']      = fft_freqs
        
        self.epsilon = 1e-9

        if extractor_kwargs['no_mean']: #Used in lfp 1D control task
            self.feature_dtype = ('ai_power', 'f8', (len(channels)*len(fft_freqs), 1))
        else: #Else: 
            self.feature_dtype = ('ai_power', 'f8', (len(channels)*len(bands), 1))

    def __call__(self, start_time, *args, **kwargs):
        cont_samples = self.get_cont_samples(*args, **kwargs)  # dims of channels x time
        #cont_samples = np.random.randn(len(self.channels), self.n_pts)  # change back!
        lfp_power = self.extract_features(cont_samples)

        return dict(ai_power=lfp_power)



    @classmethod
    def extract_from_file(cls, files, neurows, binlen, units, extractor_kwargs, strobe_rate=60.0):
        '''
        Compute binned spike count features

        Parameters
        ----------
        plx: neural data file instance
        neurows: np.ndarray of shape (T,)
            Timestamps in the plexon time reference corresponding to bin boundaries
        binlen: float
            Length of time over which to sum spikes from the specified cells
        units: np.ndarray of shape (N, 2)
            List of units that the decoder will be trained on. The first column specifies the electrode number and the second specifies the unit on the electrode
        extractor_kwargs: dict 
            Any additional parameters to be passed to the feature extractor. This function is agnostic to the actual extractor utilized
        strobe_rate: 60.0
            The rate at which the task sends the sync pulse to the plx file

        Returns
        -------
        '''
        if 'plexon' in files:
            from plexon import plexfile
            plx = plexfile.openFile(str(files['plexon']))

            # interpolate between the rows to 180 Hz
            if binlen < 1./strobe_rate:
                interp_rows = []
                neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
                for r1, r2 in zip(neurows[:-1], neurows[1:]):
                    interp_rows += list(np.linspace(r1, r2, 4)[1:])
                interp_rows = np.array(interp_rows)
            else:
                step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
                interp_rows = neurows[::step]


            # create extractor object
            f_extractor = AIMTMPowerExtractor(None, **extractor_kwargs)
            extractor_kwargs = f_extractor.extractor_kwargs

            win_len  = f_extractor.win_len
            bands    = f_extractor.bands
            channels = f_extractor.channels
            fs       = f_extractor.fs
            print(('bands:', bands))

            n_itrs = len(interp_rows)
            n_chan = len(channels)
            lfp_power = np.zeros((n_itrs, n_chan * len(bands)))
            
            # for i, t in enumerate(interp_rows):
            #     cont_samples = plx.lfp[t-win_len:t].data[:, channels-1]
            #     lfp_power[i, :] = f_extractor.extract_features(cont_samples.T).T
            lfp = plx.lfp[:].data[:, channels-1]
            n_pts = int(win_len * fs)
            for i, t in enumerate(interp_rows):
                try:
                    sample_num = int(t * fs)
                    cont_samples = lfp[sample_num-n_pts:sample_num, :]
                    lfp_power[i, :] = f_extractor.extract_features(cont_samples.T).T
                except:
                    print("Error with LFP decoder training")
                    print((i, t))
                    pass


            # TODO -- discard any channel(s) for which the log power in any frequency 
            #   bands was ever equal to -inf (i.e., power was equal to 0)
            # or, perhaps just add a small epsilon inside the log to avoid this
            # then, remember to do this:  extractor_kwargs['channels'] = channels
            #   and reset the units variable

            return lfp_power, units, extractor_kwargs

        elif 'blackrock' in files:
            raise NotImplementedError

        elif 'ecube' in files:
            raise NotImplementedError


class AIAmplitudeExtractor(object):
    '''
    Computes the analog input channel amplitude. Out of date...
    '''

    feature_type = 'ai_amplitude'

    def __init__(self, source, channels=[], win_len=0.1, fs=1000):
        self.feature_dtype = ('emg_amplitude', 'u4', (len(channels), 1))

        self.source = source
        self.channels = channels
        self.win_len = win_len
        if source is not None:
            self.fs = source.source.update_freq
        else:
            self.fs = fs

        extractor_kwargs = dict()
        extractor_kwargs['channels'] = self.channels
        extractor_kwargs['fs']       = self.fs
        extractor_kwargs['win_len']  = self.win_len
        self.extractor_kwargs = extractor_kwargs

        self.n_pts = int(self.win_len * self.fs)

    def get_cont_samples(self, *args, **kwargs):
        return self.source.get(self.n_pts, self.channels)

    def extract_features(self, cont_samples):
        n_chan = len(self.channels)
        emg_amplitude = np.mean(cont_samples,axis=1)
        emg_amplitude = emg_amplitude[:,None]
        return emg_amplitude

    def __call__(self, start_time, *args, **kwargs):
        cont_samples = self.get_cont_samples(*args, **kwargs)  # dims of channels x time
        emg = self.extract_features(cont_samples)
        return emg, None

class WaveformClusterCountExtractor(FeatureExtractor):
    feature_type = 'cluster_counts'
    def __init__(self, source, gmm_model_params, n_subbins=1, units=[]):
        self.feature_dtype = [('cluster_counts', 'f8', (len(units), n_subbins)), ('bin_edges', 'f8', 2)]

        self.source = source
        self.gmm_model_params = gmm_model_params        
        self.n_subbins = n_subbins
        self.units = units
        self.n_units = len(units)

        extractor_kwargs = dict()
        extractor_kwargs['n_subbins']        = self.n_subbins
        extractor_kwargs['units']            = self.units
        extractor_kwargs['gmm_model_params'] = gmm_model_params
        self.extractor_kwargs = extractor_kwargs

        self.last_get_spike_counts_time = 0        

    def get_spike_data(self):
        '''
        Get the spike timestamps from the neural data source. This function has no type checking, 
        i.e., it is assumed that the Extractor object was created with the proper source
        '''
        return self.source.get()

    def get_bin_edges(self, ts):
        '''
        Determine the first and last spike timestamps to allow HDF files 
        created by the BMI to be semi-synchronized with the neural data file
        '''
        if len(ts) == 0:
            bin_edges = np.array([np.nan, np.nan])
        else:
            min_ind = np.argmin(ts['ts'])
            max_ind = np.argmax(ts['ts'])
            bin_edges = np.array([ts[min_ind]['ts'], ts[max_ind]['ts']])

    def __call__(self, start_time, *args, **kwargs):

        spike_data = self.get_spike_data()
        if len(spike_data) == 0:
            counts = np.zeros([len(self.units), self.n_subbins])
        elif self.n_subbins > 1:
            subbin_edges = np.linspace(self.last_get_spike_counts_time, start_time, self.n_subbins+1)

            # Decrease the first subbin index to include any spikes that were
            # delayed in getting to the task layer due to threading issues
            # An acceptable delay is 1 sec or less. Realistically, most delays should be
            # on the millisecond order
            # subbin_edges[0] -= 1
            # subbin_inds = np.digitize(spike_data['arrival_ts'], subbin_edges)
            # counts = np.vstack([bin_spikes(ts[subbin_inds == k], self.units) for k in range(1, self.n_subbins+1)]).T
            raise NotImplementedError
        else:
            # TODO pull the waveforms
            waveforms = []

            # TODO determine p(class) for each waveform against the model params
            counts = np.zeros(self.n_units)
            wf_class_probs = []
            for wf in waveforms:
                raise NotImplementedError
            
            # counts = bin_spikes(ts, self.units).reshape(-1, 1)

        counts = np.array(counts, dtype=np.uint32)
        bin_edges = self.get_bin_edges(ts)
        self.last_get_spike_counts_time = start_time

        return dict(spike_counts=counts, bin_edges=bin_edges)        

    @classmethod
    def extract_from_file(cls, files, neurows, binlen, units, extractor_kwargs, strobe_rate=60.0):        
        from sklearn.mixture import GMM
        if 'plexon' in files:
            from plexon import plexfile
            plx = plexfile.openFile(str(files['plexon']))        
        
            channels = units[:,0]
            channels = np.unique(channels)
            np.sort(channels)

            spike_chans = plx.spikes[:].data['chan']
            spike_times = plx.spikes[:].data['ts']
            waveforms = plx.spikes[:].waveforms

            # construct the feature matrix (n_timepoints, n_units)
            # interpolate between the rows to 180 Hz
            if binlen < 1./strobe_rate:
                interp_rows = []
                neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
                for r1, r2 in zip(neurows[:-1], neurows[1:]):
                    interp_rows += list(np.linspace(r1, r2, 4)[1:])
                interp_rows = np.array(interp_rows)
            else:
                step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
                interp_rows = neurows[::step]

            # digitize the spike timestamps into interp_rows
            spike_bin_ind = np.digitize(spike_times, interp_rows)
            spike_counts = np.zeros(len(interp_rows), n_units)

            for ch in channels:
                ch_waveforms = waveforms[spike_chans == ch]

                # cluster the waveforms using a GMM
                # TODO pick the number of components in an unsupervised way!
                n_components = len(np.nonzero(units[:,0] == ch)[0])
                gmm = GMM(n_components=n_components)
                gmm.fit(ch_waveforms)

                # store the cluster probabilities back in the same order that the waveforms were extracted
                wf_probs = gmm.predict_proba(ch_waveforms)

                ch_spike_bin_inds = spike_bin_ind[spike_chans == ch]
                ch_inds, = np.nonzero(units[:,0] == ch)

                # TODO don't assume the units are sorted!
                for bin_ind, wf_prob in zip(ch_spike_bin_inds, wf_probs):
                    spike_counts[bin_ind, ch_inds] += wf_prob


            # discard units that never fired at all
            unit_inds, = np.nonzero(np.sum(spike_counts, axis=0))
            units = units[unit_inds,:]
            spike_counts = spike_counts[:, unit_inds]
            extractor_kwargs['units'] = units

            return spike_counts, units, extractor_kwargs
        elif 'blackrock' in files:
            raise NotImplementedError('Not implemented for blackrock/TDT data yet!')

        elif 'ecube' in files:
            raise NotImplementedError

def get_butter_bpf_lfp_power(plx, neurows, binlen, units, extractor_kwargs, strobe_rate=60.0):
    '''
    Compute lfp power features -- corresponds to LFPButterBPFPowerExtractor.


    '''
    
    # interpolate between the rows to 180 Hz
    if binlen < 1./strobe_rate:
        interp_rows = []
        neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
        for r1, r2 in zip(neurows[:-1], neurows[1:]):
            interp_rows += list(np.linspace(r1, r2, 4)[1:])
        interp_rows = np.array(interp_rows)
    else:
        step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
        interp_rows = neurows[::step]


    # create extractor object
    f_extractor = extractor.LFPButterBPFPowerExtractor(None, **extractor_kwargs)
    extractor_kwargs = f_extractor.extractor_kwargs

    win_len  = f_extractor.win_len
    bands    = f_extractor.bands
    channels = f_extractor.channels
    fs       = f_extractor.fs
        
    n_itrs = len(interp_rows)
    n_chan = len(channels)
    lfp_power = np.zeros((n_itrs, n_chan * len(bands)))
    # for i, t in enumerate(interp_rows):
    #     cont_samples = plx.lfp[t-win_len:t].data[:, channels-1]
    #     lfp_power[i, :] = f_extractor.extract_features(cont_samples.T).T
    lfp = plx.lfp[:].data[:, channels-1]
    n_pts = int(win_len * fs)
    for i, t in enumerate(interp_rows):
        sample_num = int(t * fs)
        cont_samples = lfp[sample_num-n_pts:sample_num, :]
        lfp_power[i, :] = f_extractor.extract_features(cont_samples.T).T
    
    # TODO -- discard any channel(s) for which the log power in any frequency 
    #   bands was ever equal to -inf (i.e., power was equal to 0)
    # or, perhaps just add a small epsilon inside the log to avoid this
    # then, remember to do this:  extractor_kwargs['channels'] = channels
    #   and reset the units variable

    return lfp_power, units, extractor_kwargs


def get_mtm_lfp_power(plx, neurows, binlen, units, extractor_kwargs, strobe_rate=60.0):
    '''
    Compute lfp power features -- corresponds to LFPMTMPowerExtractor.


    '''
    
    # interpolate between the rows to 180 Hz
    if binlen < 1./strobe_rate:
        interp_rows = []
        neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
        for r1, r2 in zip(neurows[:-1], neurows[1:]):
            interp_rows += list(np.linspace(r1, r2, 4)[1:])
        interp_rows = np.array(interp_rows)
    else:
        step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
        interp_rows = neurows[::step]


    # create extractor object
    f_extractor = extractor.LFPMTMPowerExtractor(None, **extractor_kwargs)
    extractor_kwargs = f_extractor.extractor_kwargs

    win_len  = f_extractor.win_len
    bands    = f_extractor.bands
    channels = f_extractor.channels
    fs       = f_extractor.fs
    print(('bands:', bands))

    n_itrs = len(interp_rows)
    n_chan = len(channels)
    lfp_power = np.zeros((n_itrs, n_chan * len(bands)))
    
    # for i, t in enumerate(interp_rows):
    #     cont_samples = plx.lfp[t-win_len:t].data[:, channels-1]
    #     lfp_power[i, :] = f_extractor.extract_features(cont_samples.T).T
    lfp = plx.lfp[:].data[:, channels-1]
    n_pts = int(win_len * fs)
    for i, t in enumerate(interp_rows):
        sample_num = int(t * fs)
        cont_samples = lfp[sample_num-n_pts:sample_num, :]
        lfp_power[i, :] = f_extractor.extract_features(cont_samples.T).T


    # TODO -- discard any channel(s) for which the log power in any frequency 
    #   bands was ever equal to -inf (i.e., power was equal to 0)
    # or, perhaps just add a small epsilon inside the log to avoid this
    # then, remember to do this:  extractor_kwargs['channels'] = channels
    #   and reset the units variable

    return lfp_power, units, extractor_kwargs

def get_emg_amplitude(plx, neurows, binlen, units, extractor_kwargs, strobe_rate=60.0):
    '''
    Compute EMG features.

    '''

    # interpolate between the rows to 180 Hz
    if binlen < 1./strobe_rate:
        interp_rows = []
        neurows = np.hstack([neurows[0] - 1./strobe_rate, neurows])
        for r1, r2 in zip(neurows[:-1], neurows[1:]):
            interp_rows += list(np.linspace(r1, r2, 4)[1:])
        interp_rows = np.array(interp_rows)
    else:
        step = int(binlen/(1./strobe_rate)) # Downsample kinematic data according to decoder bin length (assumes non-overlapping bins)
        interp_rows = neurows[::step]


    # create extractor object
    f_extractor = extractor.EMGAmplitudeExtractor(None, **extractor_kwargs)
    extractor_kwargs = f_extractor.extractor_kwargs

    win_len  = f_extractor.win_len
    channels = f_extractor.channels
    fs       = f_extractor.fs

    n_itrs = len(interp_rows)
    n_chan = len(channels)
    emg = np.zeros((n_itrs, n_chan))
    
    # for i, t in enumerate(interp_rows):
    #     cont_samples = plx.lfp[t-win_len:t].data[:, channels-1]
    #     lfp_power[i, :] = f_extractor.extract_features(cont_samples.T).T
    emgraw = plx.analog[:].data[:, channels-1]
    n_pts = int(win_len * fs)
    for i, t in enumerate(interp_rows):
        sample_num = int(t * fs)
        cont_samples = emgraw[sample_num-n_pts:sample_num, :]
        emg[i, :] = f_extractor.extract_features(cont_samples.T).T

    return emg, units, extractor_kwargs
