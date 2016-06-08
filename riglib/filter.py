'''
Generic class for implementing filters describable by rational z-transforms (ratio of polynomials)
'''
from scipy.signal import sigtools, lfilter
import numpy as np

class Filter(object):
    def __init__(self, b=[], a=[1.]):
        '''
        Constructor for Filter

        Parameters
        ----------
        b : array_like
            The numerator coefficient vector in a 1-D sequence.
        a : array_like
            The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
            is not 1, then both `a` and `b` are normalized by ``a[0]``.

        Returns
        -------
        Filter instance
        '''
        self.b = np.array(b)
        self.a = np.array(a)

        # normalize the constants
        self.b /= self.a[0]
        self.a /= self.a[0]

        self.zi = np.zeros(max(len(a), len(b))-1)


    def __call__(self, samples):
        '''
        Run the filter parameters on the most recent set of samples

        Parameters
        ----------
        samples : np.ndarray of shape (N,)
            samples to filter (x input)

        Returns
        -------
        np.ndarray
            Most recent N outputs of the filter
        '''
        # promote scalars to arrays so the lfilter function doesn't complain
        if isinstance(samples, (float, int)):
            samples = np.array([samples])

        filt_output, self.zi = lfilter(self.b, self.a, samples, zi=self.zi)
        return filt_output