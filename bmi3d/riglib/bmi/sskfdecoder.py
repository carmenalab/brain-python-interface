'''
Classes for BMI decoding using the Steady-state Kalman filter. Based
heavily on the kfdecoder module. 
'''
import numpy as np
from . import bmi
from . import kfdecoder

class SteadyStateKalmanFilter(bmi.GaussianStateHMM):
    """
    Low-level KF in steady-state

    Model: 
       x_{t+1} = Ax_t + w_t;   w_t ~ N(0, W)
           y_t = Cx_t + q_t;   q_t ~ N(0, Q)
    """ 
    model_attrs = ['F', 'K']

    def __init__(self, *args, **kwargs):
        '''Docstring

        Parameters
        ----------

        Returns
        -------

        '''
        if len(list(kwargs.keys())) == 0:
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


    def _init_state(self, init_state=None, init_cov=None):                     
        """                                                                    
        Initialize the state of the filter with a mean and covariance (uncertainty)
        Docstring

        Parameters
        ----------

        Returns
        -------
        """                                                                    
        ## Initialize the BMI state, assuming                                  
        nS = self.n_states                                                 
        if init_state == None:                                                 
            init_state = np.mat( np.zeros([nS, 1]) )                           
            if self.include_offset: init_state[-1,0] = 1                       
        if init_cov == None:                                                   
            init_cov = np.mat( np.zeros([nS, nS]) )
        self.state = bmi.GaussianState(init_state, init_cov)                       

    def _pickle_init(self):
        '''Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        nS = self.F.shape[0]
        self.I = np.mat(np.eye(nS))

    def get_sskf(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return self.F, self.K

    def _forward_infer(self, st, obs_t, Bu=None, u=None, target_state=None, 
                       obs_is_control_independent=True, bias_comp=False, **kwargs):
        '''
        Estimate p(x_t | ..., y_{t-1}, y_t)
        Docstring

        Parameters
        ----------

        Returns
        -------
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
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return dict(F=self.F, K=self.K)
    
    @property
    def n_states(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return self.F.shape[0]

    @property
    def include_offset(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return np.all(self.F[-1, :-1] == 0) and (self.F[-1, -1] == 1)

    def get_K_null(self):
        '''
        $$y_{null} = K_{null} * y_t$$ gives the "null" component of the spike inputs, i.e. $$K_t*y_{null} = 0_{N\times 1}$$
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        K = np.mat(self.K)
        n_neurons = K.shape[1]
        K_null = np.eye(n_neurons) - np.linalg.pinv(K) * K
        return K_null

class SSKFDecoder(bmi.Decoder, bmi.BMI):
    ''' Docstring '''
    pass
