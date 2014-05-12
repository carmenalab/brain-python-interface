'''
Classes for BMI decoding using the Steady-state Kalman filter. Based
heavily on the kfdecoder module. 
'''
import numpy as np
from scipy.io import loadmat

import bmi
import train
import pickle

import kfdecoder

class SteadyStateKalmanFilter(bmi.GaussianStateHMM):
    """
    Low-level KF in steady-state

    Model: 
       x_{t+1} = Ax_t + w_t;   w_t ~ N(0, W)
           y_t = Cx_t + q_t;   q_t ~ N(0, Q)
    """ 
    model_attrs = ['F', 'K']

    def __init__(self, *args, **kwargs):
        if len(kwargs.keys()) == 0:
            ## This condition should only be true in the unpickling phase
            pass
        else:
            if 'A' in kwargs and 'C' in kwargs and 'W' in kwargs and 'Q' in kwargs:
                A = kwargs.pop('A')
                C = kwargs.pop('C')
                W = kwargs.pop('W')
                Q = kwargs.pop('Q')
                kf = kfdecoder.KalmanFilter(A, W, C, Q, **kwargs)
                F, K = kf.get_sskf()
            elif 'F' in kwargs and 'K' in kwargs:
                F = kwargs['F']
                K = kwargs['K']
            self.F = F
            self.K = K
            self._pickle_init()

    def _pickle_init(self):
        nS = self.F.shape[0]
        self.I = np.mat(np.eye(nS))

    def get_sskf(self):
        return self.F, self.K

    def _forward_infer(self, st, obs_t, Bu=None, u=None, target_state=None, 
                       obs_is_control_independent=True, bias_comp=False):
        '''
        Estimate p(x_t | ..., y_{t-1}, y_t)
        '''
        F, K = self.F, self.K
        if Bu is not None:
            post_state_mean = F*st.mean + K*obs_t + Bu
        else:
            post_state_mean = F*st.mean + K*obs_t

        I = self.I
        post_state = I*st # Force memory reallocation for the Gaussian
        post_state.mean = post_state_mean
        return post_state

    def __getstate__(self):
        '''
        Pickle only the F and the K matrices
        '''
        return dict(F=self.F, K=self.K)

class SSKFDecoder(bmi.Decoder, bmi.BMI):
    pass