'''
Classes for BMI decoding using the Wiener filter (linear filter on a history of neural features). 
'''

import numpy as np

from . import bmi

class WienerFilter(bmi.GaussianStateHMM):
    """
    Low-level Wiener filter, agnostic to application

    Model: 
       x_{t+1} = Ax_t + w_t;   w_t ~ N(0, W)
           x_t = H*[y_t; y_{t-1}; ...; y_{t-L+1}; 1]

    The states which are estimated from the observations (specified by 'is_stochastic') 
    are a linear function of the last L = 'n_taps' observations, i.e., an FIR filter on 
    the neural features. All the other states are propagated using the state transition 
    model only, e.g., position states are updated by integrating the decoded velocity. 
    """
    model_attrs = ['A', 'W', 'H']
    attrs_to_pickle = ['A', 'W', 'H', 'n_taps', 'is_stochastic', 'B', 'F']

    def __init__(self, A=None, W=None, H=None, n_taps=1, is_stochastic=None, B=None, F=None):
        '''
        Constructor for WienerFilter

        Parameters
        ----------
        A : np.mat, optional
            Model of state transition matrix
        W : np.mat, optional
            Model of process noise covariance
        H : np.mat, optional
            Filter weights mapping the observation history onto the state, of shape 
            (n_states, n_features*n_taps + 1). The last column is the offset term and 
            the rows of states which are not estimated from the observations are ignored
        n_taps : int, optional
            Number of observations (including the current one) which the filter acts on.
            Default is 1, i.e., the state is a linear function of the current observation only
        is_stochastic : np.array, optional
            Array of booleans specifying for each state whether it is estimated from the 
            observations. If 'None' specified, all states are assumed to be stochastic
        B : np.mat, optional
            Control input matrix, required for the 'u' and 'x_target' control inputs
        F : np.mat, optional
            Default feedback controller gains, used with 'x_target' if no gains are given

        Returns
        -------
        WienerFilter instance
        '''
        if A is None and W is None and H is None:
            ## This condition should only be true in the unpickling phase
            pass
        else:
            self.A = np.asmatrix(A)
            self.W = np.asmatrix(W)
            self.H = np.asmatrix(H)
            self.n_taps = int(n_taps)
            self.B = None if B is None else np.asmatrix(B)
            self.F = None if F is None else np.asmatrix(F)

            if is_stochastic is None:
                n_states = self.A.shape[0]
                self.is_stochastic = np.ones(n_states, dtype=bool)
            else:
                self.is_stochastic = np.array(is_stochastic)

            self.state_noise = bmi.GaussianState(0.0, self.W)
            self._pickle_init()

    def _pickle_init(self):
        """Code common to unpickling and initialization
        """
        nS = self.A.shape[0]
        offset_row = np.zeros(nS)
        offset_row[-1] = 1
        self.include_offset = np.array_equal(np.array(self.A)[-1, :], offset_row)

        if not hasattr(self, 'n_taps'):
            self.n_taps = 1
        if not hasattr(self, 'B'):
            self.B = None
        if not hasattr(self, 'F'):
            self.F = None

        try:
            self.is_stochastic
        except:
            self.is_stochastic = np.ones(nS, dtype=bool)

        # the last column of H is the offset term
        self.n_features = (self.H.shape[1] - 1) // self.n_taps

    def init_noise_models(self):
        '''
        see bmi.GaussianStateHMM.init_noise_models for documentation. The Wiener filter 
        has no observation noise model, as the observations are never predicted from the state
        '''
        self.state_noise = bmi.GaussianState(0.0, self.W)

    def _init_state(self, init_state=None, init_cov=None):
        '''
        Initialize the state of the filter and clear the history of observations. 
        See bmi.GaussianStateHMM._init_state for documentation
        '''
        super(WienerFilter, self)._init_state(init_state=init_state, init_cov=init_cov)
        self.obs_hist = np.zeros([self.n_taps, self.n_features])

    def _add_obs(self, obs_t):
        '''
        Shift the new observation into the history of observations

        Parameters
        ----------
        obs_t : np.mat of shape (N, 1)
            Neural features observed at the current time step

        Returns
        -------
        None
        '''
        self.obs_hist[1:, :] = self.obs_hist[:-1, :]
        self.obs_hist[0, :] = np.asarray(obs_t).ravel()

    def _get_obs_vec(self):
        '''
        Stack the history of observations into the regressor vector the filter acts on, 
        i.e., [y_t; y_{t-1}; ...; y_{t-L+1}; 1]

        Parameters
        ----------
        None

        Returns
        -------
        np.mat of shape (n_features*n_taps + 1, 1)
        '''
        return np.asmatrix(np.hstack([self.obs_hist.ravel(), 1.])).T

    def _forward_infer(self, st, obs_t, Bu=None, u=None, x_target=None, F=None, **kwargs):
        '''
        Estimate p(x_t | ..., y_{t-1}, y_t)

        Parameters
        ----------
        st : GaussianState
            Current estimate (mean and cov) of hidden state
        obs_t : np.mat of shape (N, 1)
            Neural features observed at the current time step
        Bu : np.mat of shape (N, 1), optional, default=None
            Assistive control input which already accounts for the control input matrix
        u : np.mat, optional, default=None
            Assistive control input
        x_target : np.mat of shape (N, 1), optional, default=None
            Optimal state, used with the feedback controller gains 'F'
        F : np.mat, optional, default=None
            Feedback controller gains
        kwargs : optional kwargs
            Ignored, for compatibility with the other filters

        Returns
        -------
        GaussianState
            New state estimate incorporating the most recent observation
        '''
        if Bu is None and (u is not None or x_target is not None):
            if self.B is None:
                raise ValueError("WienerFilter has no control input matrix B, so 'u' and 'x_target' are not supported. Use 'Bu' instead")
            if u is None and F is None and self.F is None:
                raise ValueError("WienerFilter has no feedback controller gains F, so they must be supplied with 'x_target'")

        obs_t = np.asmatrix(np.asarray(obs_t).reshape(-1, 1))
        self._add_obs(obs_t)

        # states which are not estimated from the observations are propagated by the SSM,
        # e.g., position is updated by integrating the previously decoded velocity
        post_state = self._ssm_pred(st, target_state=x_target, Bu=Bu, u=u, F=F)

        # the estimate is independent of the prior, so the control input has to be added
        # back on to the states which the filter estimates
        c_t = post_state.mean - self.A * st.mean

        inds, = np.nonzero(self.is_stochastic)
        post_state.mean[inds, :] = self.H[inds, :] * self._get_obs_vec() + c_t[inds, :]

        return post_state

    @classmethod
    def form_obs_history(self, obs, n_taps=1, include_offset=True):
        """
        Stack time-lagged copies of the observations, to form the matrix of regressors 
        which the filter weights act on

        Parameters
        ----------
        obs : np.ndarray of shape (N, T)
            N = number of features, T = number of observations
        n_taps : int, optional, default=1
            Number of observations (including the current one) which the filter acts on
        include_offset : bool, optional, default=True
            A row of all 1's is added as the last row to estimate an offset term

        Returns
        -------
        np.ndarray of shape (N*n_taps + 1, T)
            Column 't' is [y_t; y_{t-1}; ...; y_{t-L+1}; 1]. The first (n_taps - 1) columns 
            are zero-padded, matching the way the filter starts up online
        """
        n_features, T = obs.shape
        obs_hist = np.zeros([n_features*n_taps, T])
        for k in range(n_taps):
            obs_hist[k*n_features:(k+1)*n_features, k:] = obs[:, :T-k]

        if include_offset:
            obs_hist = np.vstack([obs_hist, np.ones([1, T])])
        return obs_hist

    @classmethod
    def MLE_filter(self, hidden_state, obs, n_taps=1, include_offset=True, regularizer=None):
        """
        Least-squares estimate of the filter weights H given observations and the 
        corresponding hidden states, i.e., the solution of the Wiener-Hopf equations

        Parameters
        ----------
        hidden_state : np.ndarray of shape (N, T)
            N = dimensionality of state vector, T = number of observations
        obs : np.ndarray of shape (M, T)
            M = number of features, T = number of observations
        n_taps : int, optional, default=1
            Number of observations (including the current one) which the filter acts on
        include_offset : bool, optional, default=True
            Estimate an offset term in addition to the filter weights. If False, the 
            offset column of H is zero
        regularizer : float, optional, default=None
            Ridge penalty on the filter weights. The offset term is not penalized

        Returns
        -------
        H : np.mat of shape (N, M*n_taps + 1)
            Filter weights mapping the observation history onto the hidden state. The last
            column is always the offset term, as expected by WienerFilter
        """
        assert hidden_state.shape[1] == obs.shape[1], "different numbers of time samples: %s vs %s" % (str(hidden_state.shape), str(obs.shape))

        Y = self.form_obs_history(np.asarray(obs), n_taps=n_taps, include_offset=include_offset)
        X = np.asarray(hidden_state)

        # discard the samples for which the history of observations is incomplete
        Y = Y[:, n_taps-1:]
        X = X[:, n_taps-1:]

        if regularizer is None:
            H = np.linalg.lstsq(Y.T, X.T, rcond=None)[0].T
        else:
            penalty = regularizer * np.eye(Y.shape[0])
            if include_offset:
                penalty[-1, -1] = 0 # don't penalize the offset term
            YtY_lamb = Y.dot(Y.T) + penalty
            YtX = Y.dot(X.T)
            H = np.linalg.solve(YtY_lamb, YtX).T

        if not include_offset:
            H = np.hstack([H, np.zeros((H.shape[0], 1))])

        return np.asmatrix(H)

class WFDecoder(bmi.BMI, bmi.Decoder):
    '''
    Wrapper for WienerFilter specifically for the application of BMI decoding.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for WFDecoder

        Parameters
        ----------
        *args, **kwargs : see riglib.bmi.bmi.Decoder for arguments

        Returns
        -------
        WFDecoder instance
        '''
        mFR = kwargs.pop('mFR', 0.)
        sdFR = kwargs.pop('sdFR', 1.)

        super(WFDecoder, self).__init__(*args, **kwargs)
        self.mFR = mFR
        self.sdFR = sdFR
        self.zeromeanunits = None
        self.zscore = False
        self.wf = self.filt

    def __setstate__(self, state):
        """
        Set decoder state after un-pickling. See Decoder.__setstate__, which runs the _pickle_init function at some point during the un-pickling process

        Parameters
        ----------
        state : dict
            Variables to set as attributes of the unpickled object.

        Returns
        -------
        None
        """
        if 'wf' in state and 'filt' not in state:
            state['filt'] = state['wf']

        super(WFDecoder, self).__setstate__(state)
        self.wf = self.filt

    @property
    def n_taps(self):
        '''
        Number of time-lagged observations the decoder acts on
        '''
        return self.filt.n_taps

    def plot_H(self, tap=0, **kwargs):
        '''
        Plot the filter weights for a single tap

        Parameters
        ----------
        tap : int, optional, default=0
            Which lag of the filter to plot, 0 being the most recent observation
        **kwargs : optional kwargs
            These are passed to the plot function (e.g., which rows to plot)

        Returns
        -------
        None
        '''
        n_features = self.filt.n_features
        H = self.filt.H[:, tap*n_features:(tap+1)*n_features]
        self.plot_pds(H.T, **kwargs)
