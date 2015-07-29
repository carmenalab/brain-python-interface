'''
Code for feature extraction methods/classes from EMG, to be used with a 
decoder (similar to other types of feature extractors in riglib.bmi.extractor)
'''

from collections import OrderedDict
from scipy.signal import butter, lfilter
import numpy as np


def extract_MAV(samples):
    '''
    Calculate the mean absolute value (MAV) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    return np.mean(abs(samples), axis=1, keepdims=True)

def extract_WAMP(samples, threshold=0):
    '''
    Calculate the Willison amplitude (WAMP) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts
    threshold : np.float
        threshold in uV below which WAMP isn't counted

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    condition = abs(np.diff(samples)) >= threshold
    return np.sum(condition, axis=1, keepdims=True)

def extract_VAR(samples):
    '''
    Calculate the variance (VAR) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    N = samples.shape[1]
    return (1./(N-1)) * np.sum(samples**2, axis=1, keepdims=True)

def extract_WL(samples):
    '''
    Calculate the waveform length (WL) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    return np.sum(abs(np.diff(samples)), axis=1, keepdims=True)

def extract_RMS(samples):
    '''
    Calculate the root mean square (RMS) value for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    N = samples.shape[1]
    return np.sqrt((1./N) * np.sum(samples**2, axis=1, keepdims=True))

def extract_ZC(samples, threshold=0):
    '''
    Compute the number of zero crossings (ZC) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts
    threshold : np.float
        threshold in uV below which zero crossings aren't counted

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    zero_crossing = np.sign(samples[:, 1:] * samples[:, :-1]) == -1
    greater_than_threshold = abs(np.diff(samples)) >= threshold
    condition = np.logical_and(zero_crossing, greater_than_threshold)

    return np.sum(condition, axis=1, keepdims=True)

def extract_SSC(samples, threshold=0):
    '''
    Compute the number of slope-sign changes (SSC) for multiple channels.

    Parameters
    ----------
    samples : np.ndarray of shape (n_channels, n_time_points)
        Observed EMG voltages in microvolts
    threshold : np.float
        threshold in uV below which SSCs aren't counted

    Returns
    -------
    np.ndarray of shape (n_channels, 1)
    '''

    diff = np.diff(samples)
    condition = diff[:, 1:] * diff[:, :-1] >= threshold
    return np.sum(condition, axis=1, keepdims=True)


# dictionary mapping feature names to the corresponding functions defined above 
FEATURE_FUNCTIONS_DICT = {
    'MAV':  extract_MAV,
    'WAMP': extract_WAMP,
    'VAR':  extract_VAR,
    'WL':   extract_WL,
    'RMS':  extract_RMS,
    'ZC':   extract_ZC,
    'SSC':  extract_SSC,
}

from riglib.bmi.extractor import FeatureExtractor
class EMGMultiFeatureExtractor(FeatureExtractor):
    '''
    Extract many different types of EMG features from raw EMG voltages
    '''

    feature_type = 'emg_multi_features'

    def __init__(self, source=None, channels=[], feature_names=FEATURE_FUNCTIONS_DICT.keys(), feature_fn_kwargs={}, win_len=0.2, fs=1000):  
        '''
        Constructor for EMGMultiFeatureExtractor

        Parameters
        ----------
        source : MultiChanDataSource instance, optional, default=None
            DataSource interface to separate process responsible for collecting data from the EMG recording system
        channels : iterable of strings, optional, default=[]
            Names of channels from which to extract data
        feature_names : iterable, optional, default=[]
            Types of features to include in the extractor's output. See FEATURE_FUNCTIONS_DICT for available options
        feature_fn_kwargs : dict, optional, default={}
            Optional kwargs to pass to the individual feature extractors
        win_len : float, optional, default=0.2
            Length of time (in seconds) of raw EMG data to use for feature extraction
        fs : float, optional, default=1000
            Sampling rate for the EMG data

        Returns
        -------
        EMGMultiFeatureExtractor instance
        '''

        self.source            = source
        self.channels          = channels
        self.feature_names     = feature_names
        self.feature_fn_kwargs = feature_fn_kwargs
        self.win_len           = win_len
        if source is not None:
            self.fs = source.source.update_freq
        else:
            self.fs = fs

        self.n_features = len(channels) * len(feature_names)
        self.feature_dtype = ('emg_multi_features', 'u4', self.n_features, 1)

        self.n_win_pts = int(self.win_len * self.fs)

        # calculate coefficients for a 4th-order Butterworth BPF from 10-450 Hz
        band  = [10, 450]  # Hz
        nyq   = 0.5 * self.fs
        low   = band[0] / nyq
        high  = band[1] / nyq
        self.bpf_coeffs = butter(4, [low, high], btype='band')

        # calculate coefficients for multiple 2nd-order notch filers
        self.notchf_coeffs = []
        for freq in [50, 150, 250, 350]:
            band  = [freq - 1, freq + 1]  # Hz
            nyq   = 0.5 * self.fs
            low   = band[0] / nyq
            high  = band[1] / nyq
            self.notchf_coeffs.append(butter(2, [low, high], btype='bandstop'))
        
    def get_samples(self):
        '''
        Get samples from this extractor's MultiChanDataSource.

        Parameters
        ----------
        None 

        Returns
        -------
        Voltage samples of shape (n_channels, n_time_points)
        '''
        return self.source.get(self.n_win_pts, self.channels)['data']

    def extract_features(self, samples):
        '''
        Parameters
        ----------
        samples : np.ndarray of shape (n_channels, n_time_points)
            Raw EMG voltages from which to extract features

        Returns
        -------
        features : np.ndarray of shape (n_features, 1)
        '''

        # apply band-pass filter
        b, a = self.bpf_coeffs
        samples = lfilter(b, a, samples)

        # apply notch filters
        for b, a in self.notchf_coeffs:
            samples = lfilter(b, a, samples)

        # extract actual features
        features = np.zeros((0, 1))
        for name in self.feature_names:
            fn = FEATURE_FUNCTIONS_DICT[name]
            try:
                kwargs = self.feature_fn_kwargs[name]
            except KeyError:
                kwargs = {}  # empty dictionary of kwargs
            new_features = fn(samples, **kwargs)
            features = np.vstack([features, new_features])

        return features.reshape(-1)

    def __call__(self):
        '''
        Get samples from this extractor's data source and extract features.
        '''
        samples  = self.get_samples()
        features = self.extract_features(samples)
        return dict(emg_multi_features=features)


class ReplayEMGMultiFeatureExtractor(EMGMultiFeatureExtractor):
    '''
    Extract EMG features from EMG data stored in a file (instead of reading from the streaming input source)
    '''
    def __init__(self, hdf_table=None, cycle_rate=60., **kwargs):
        '''
        Parameters
        ----------
        hdf_table : HDF table
            Data table to replay, e.g., hdf.root.brainamp
        cycle_rate : float, optional, default=60.0
            Rate at which the task FSM "cycles", i.e., the rate at which the task will ask for new observations
        '''
        kwargs.pop('source', None)
        super(ReplayEMGMultiFeatureExtractor, self).__init__(source=None, **kwargs)
        self.hdf_table = hdf_table
        self.n_calls = 0
        self.cycle_rate = cycle_rate

    def get_samples(self):
        self.n_calls += 1
        table_idx = int(1./self.cycle_rate * self.n_calls * self.fs)
        table_idx = max(table_idx, 1)
        # import pdb; pdb.set_trace()
        start_idx = max(table_idx - self.n_win_pts, 0)
        if 0:
            print "self.channels"
            print self.channels
            for ch in self.channels:
                print ch
                print self.hdf_table[:table_idx][ch]['data']
        samples = np.vstack([self.hdf_table[:table_idx][ch]['data'] for ch in self.channels])
        return samples
