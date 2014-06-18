'''
Feature accumulators: The task and the decoder may want to run at
different rates. These modules provide rate-matching
'''
import numpy as np

class RectWindowSpikeRateEstimator(object):
    '''    Docstring    '''
    def __init__(self, count_max, feature_shape, feature_dtype):
        '''    Docstring    '''
        self.count_max = count_max
        self.feature_shape = feature_shape
        self.feature_dtype = feature_dtype
        self.reset()

    def reset(self):
        '''    Docstring    '''
        self.est = np.zeros(self.feature_shape, dtype=self.feature_dtype)
        self.count = 0

    def __call__(self, features):
        '''    Docstring    '''
        self.count += 1
        self.est += features
        est = self.est
        if self.count == self.count_max:
            est = self.est.copy()
            self.reset()
        return est


class NullAccumulator(object):
    '''    Docstring    '''
    def __init__(*args, **kwargs):
        '''    Docstring    '''
        pass

    def __call__(self, features):
        '''    Docstring    '''
        return features