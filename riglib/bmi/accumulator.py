'''
Feature accumulators: The task and the decoder may want to run at
different rates. These modules provide rate-matching
'''
import numpy as np

class FeatureAccumulator(object):
    '''Used only for type-checking'''
    pass

class RectWindowSpikeRateEstimator(FeatureAccumulator):
    '''
    Estimate spike firing rates using a rectangular window
    '''
    def __init__(self, count_max, feature_shape, feature_dtype):
        '''
        Constructor for RectWindowSpikeRateEstimator

        Parameters
        ----------
        count_max : int 
            Number of bins to accumulate in the window. This is somewhat specific
            to rectangular binning
        feature_shape : np.array of shape (n_features, n_timepoints)
            Shape of the extracted features passed to the Decoder on each call
        feature_dtype : np.dtype
            Data type of feature vector. Can be "np.float64" for a vector of numbers
            or something more complicated.

        Returns
        -------
        RectWindowSpikeRateEstimator instance
        '''
        self.count_max = count_max
        self.feature_shape = feature_shape
        self.feature_dtype = feature_dtype
        self.reset()

    def reset(self):
        '''
        Reset the current estimate of the spike rates. Used at the end of the window to clear the estimator for the new window
        '''
        self.est = np.zeros(self.feature_shape, dtype=self.feature_dtype)
        self.count = 0

    def __call__(self, features):
        '''
        Accumulate the current 'features' with the previous estimate

        Parameters
        ----------
        features: np.ndarray of shape self.features_shape 
            self.feature_shape is declared at object creation time

        Returns
        -------
        est: np.ndarray of shape self.features_shape
            Returns current estimate of features. This estimate may or may not be 
            valid depending on when the estimate is checked

        '''
        self.count += 1
        self.est += features
        est = self.est
        decode = False
        if self.count == self.count_max:
            est = self.est.copy()
            self.reset()
            decode = True
        return est, decode

class NullAccumulator(FeatureAccumulator):
    '''
    A null accumulator to use in cases when no accumulation is desired.
    '''
    def __init__(self, count_max):
        '''
        Constructor for NullAccumulator

        Parameters
        ----------
        count_max: int 
            Number of bins to accumulate in the window. This is somewhat specific
            to rectangular binning

        Returns
        -------
        NullAccumulator instance
        '''
        self.count_max = count_max
        self.reset()

    def reset(self):
        '''
        Reset the counter
        '''
        self.count = 0

    def __call__(self, features):
        '''
        Accumulate the current 'features' with the previous estimate

        Parameters
        ----------
        features: np.ndarray of shape self.features_shape 
            self.feature_shape is declared at object creation time

        Returns
        -------
        est: np.ndarray of shape self.features_shape
            Returns current estimate of features. This estimate may or may not be 
            valid depending on when the estimate is checked
        '''
        self.count += 1
        decode = False
        if self.count == self.count_max:
            self.reset()
            decode = True
        return features, decode