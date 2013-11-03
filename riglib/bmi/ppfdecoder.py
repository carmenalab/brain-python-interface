'''Needs docs'''

import numpy as np

import bmi
from bmi import GaussianState
import statsmodels.api as sm # GLM fitting module
import time
import cmath

class PointProcessFilter():
    """
    Low-level Point-process filter, agnostic to application

    Model: 
       x_{t+1} = Ax_t + w_t; w_t ~ N(0, W)
       log(y_t) = Cx_t
    """

    def __init__(self, A, W, C, dt, is_stochastic=None, B=0, F=0):
        self.A = np.mat(A)
        self.W = np.mat(W)
        self.C = np.mat(C)
        self.dt = dt
        
        self.B = B
        self.F = F

        if is_stochastic == None:
            n_states = A.shape[0]
            self.is_stochastic = np.ones(n_states, dtype=bool)
        else:
            self.is_stochastic = is_stochastic
        
        self.state_noise = GaussianState(0.0, W)
        self._pickle_init()

    def _pickle_init(self):
        """Code common to unpickling and initialization
        """
        nS = self.A.shape[0]
        offset_row = np.zeros(nS)
        offset_row[-1] = 1
        self.include_offset = np.array_equal(np.array(self.A)[-1, :], offset_row)

        if not hasattr(self, 'B'): self.B = 0
        if not hasattr(self, 'F'): self.F = 0

    def _init_state(self, init_state=None, init_cov=None):
        """ Initialize the state of the KF prior to running in real-time
        """
        ## Initialize the BMI state, assuming 
        nS = self.A.shape[0] # number of state variables
        if init_state == None:
            init_state = np.mat( np.zeros([nS, 1]) )
            if self.include_offset: init_state[-1,0] = 1
        if init_cov == None:
            init_cov = np.mat( np.zeros([nS, nS]) )
        self.state = GaussianState(init_state, init_cov) 
        self.state_noise = GaussianState(0.0, self.W)
        self.id = np.zeros([1, self.C.shape[0]])

    def __call__(self, obs, target_state=None):
        """ Call the 1-step forward inference function
        """
        self.state = self._forward_infer(self.state, obs, target_state=target_state)
        return self.state.mean

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def _check_valid(self, lambda_predict):
        if np.any((lambda_predict * self.dt) > 1): 
            raise ValueError("Cell exploded!")

    def _obs_prob(self, state):
        Loglambda_predict = self.C * state.mean
        lambda_predict = np.exp(Loglambda_predict)/self.dt
        assert np.all(lambda_predict > 0)

        #self._check_valid(self, lambda_predict, id)
        return lambda_predict
    
    def _ssm_pred(self, state, target_state=None):
        A = self.A
        B = self.B
        F = self.F
        if target_state == None:
            return A*state + self.state_noise
        else:
            return (A - B*F)*state + B*F*target_state + self.state_noise

    def _forward_infer(self, st, obs_t, target_state=None): #stimulant_index, stoch_stim_index, stoch_index, det_index)
        obs_t = np.mat(obs_t.reshape(-1,1))
        C = self.C
        n_obs, n_states = C.shape
        
        dt = self.dt
        inds, = np.nonzero(self.is_stochastic)
        mesh = np.ix_(inds, inds)
        A = self.A#[mesh]
        W = self.W#[mesh]
        C = C[:,inds]

        x_prev, P_prev = st.mean, st.cov
        #x_prev = x_prev[inds,:]
        #P_prev = P_prev[mesh]
        x_pred = A*x_prev
        P_pred = A*P_prev*A.T + W
        P_pred = P_pred[mesh]

        Loglambda_predict = self.C * x_pred #self.C[:,-1] + C * x_pred #self.C * 
        exp = np.vectorize(lambda x: np.real(cmath.exp(x)))
        lambda_predict = exp(np.array(Loglambda_predict).ravel())/dt

        Q_inv = np.mat(np.diag(lambda_predict*self.dt))

        if np.linalg.cond(P_pred) > 1e5:
            P_est = P_pred;
        else:
            P_est = (P_pred.I + C.T*np.mat(np.diag(lambda_predict*dt))*C).I

        # inflate P_est
        P_est_full = np.mat(np.zeros([n_states, n_states]))
        P_est_full[mesh] = P_est
        P_est = P_est_full 

        unpred_spikes = obs_t - np.mat(lambda_predict*dt).reshape(-1,1)

        # TODO fix indexing
        x_est = np.mat(np.zeros([n_states,1]))
        x_est = x_pred + P_est*self.C.T*unpred_spikes
        #x_est[-1,0] = 1 # offset state
        post_state = GaussianState(x_est, P_est)
        #assert post_state.mean.shape == (3,1)
        #import pdb; pdb.set_trace()

        return post_state

        #### pred_state = self._ssm_pred(st, target_state=target_state)
        #### pred_obs = self._obs_prob(pred_state)
        #### #print pred_obs

        #### P_pred = pred_state.cov
        #### inds, = np.nonzero(self.is_stochastic)
        #### nS = self.A.shape[0]

        #### P_pred_inv = np.mat(np.zeros([nS, nS]))
        #### mesh = np.ix_(inds, inds)
        #### P_pred_inv[mesh] = P_pred[mesh].I
        #### #P_pred_inv[:-1, :-1] = P_pred[:-1,:-1].I
        #### P_est = np.mat(np.zeros([nS, nS]))
        #### P_est[mesh] = (P_pred_inv[mesh] + C[:,inds].T*Q_inv*C[:,inds]).I

        ## New version, deprecated

        #### if n_obs > n_states:
        ####     Q_inv = np.mat(np.diag(np.array(pred_obs).ravel() * self.dt))
        ####     I = np.mat(np.eye(nS))
        ####     D = C.T * Q_inv * C
        ####     ### P_est = P_pred - P_pred*((I - D*P_pred*(I + D*P_pred).I)*D)*P_pred
        ####     ### I = np.mat(np.eye(n_obs))
        ####     ### P_est = P_pred - P_pred*C.T*Q_inv * (I + C * P_pred * C.T*Q_inv).I * C * P_pred
        ####     
        ####     #F = C.T * (Q_inv.I + C*P_pred*C.T).I * C
        ####     # ... after mat inv lemma:
        ####     #F = C.T * (Q_inv - Q_inv*C*P_pred*(I + D).I * C.T*Q_inv) * C
        ####     # distr
        ####     #F = (C.T *Q_inv * C - C.T *Q_inv*C*P_pred*(I + D).I * C.T*Q_inv * C)
        ####     # sub
        ####     F = (D - D*P_pred*(I + D).I * D)
        ####     P_est = P_pred - P_pred * F * P_pred
        #### elif n_obs == 1:
        ####     if isinstance(pred_obs, np.ndarray) or isinstance(pred_obs, np.matrix):
        ####         pred_obs = pred_obs[0,0]
        ####     q = 1./(pred_obs*self.dt)
        ####     P_est = P_pred - 1./q * (P_pred*C.T)*(C*P_pred)
        #### else:
        ####     #Q = Q_inv.I # TODO zero out diagonal if any pred are 0 (occurs w.p. 0...)
        ####     Q_diag = (np.array(pred_obs).ravel() * self.dt)**-1
        ####     Q = np.mat(np.diag(Q_diag))
        ####     P_est = P_pred - P_pred*C.T * (Q + C*P_pred*C.T).I * C*P_pred


    def __setstate__(self, state):
        """Set the model parameters {A, W, C, Q} stored in the pickled
        object"""
        # TODO clean this up!
        self.A = state['A']
        self.W = state['W']
        self.C = state['C']
        self.B = state['B']
        self.is_stochastic = state['is_stochastic']
        self.dt = state['dt']
        self._pickle_init()

    def __getstate__(self):
        """Return the model parameters {A, W, C} for pickling"""
        return dict(A=self.A, W=self.W, C=self.C, dt=self.dt, B=self.B, 
                    is_stochastic=self.is_stochastic)
        #return {'A':self.A, 'W':self.W, 'C':self.C, 'dt':self.dt, }

    @classmethod
    def MLE_obs_model(cls, hidden_state, obs, include_offset=True, drives_obs=None):
        """
        Unconstrained ML estimator of {C, } given observations and
        the corresponding hidden states
        """
        assert hidden_state.shape[1] == obs.shape[1]
    
        if isinstance(hidden_state, np.ma.core.MaskedArray):
            mask = ~hidden_state.mask[0,:] # NOTE THE INVERTER 
            inds = np.nonzero([ mask[k]*mask[k+1] for k in range(len(mask)-1)])[0]
    
            X = np.mat(hidden_state[:,mask])
            T = len(np.nonzero(mask)[0])
    
            Y = np.mat(obs[:,mask])
            if include_offset:
                X = np.vstack([ X, np.ones([1,T]) ])
        else:
            num_hidden_state, T = hidden_state.shape
            X = np.mat(hidden_state)
            if include_offset:
                X = np.vstack([ X, np.ones([1,T]) ])
                if not drives_obs == None:
                    drives_obs = np.hstack([drives_obs, True])
                
            Y = np.mat(obs)
        
        X = np.array(X)
        if not drives_obs == None:
            X = X[drives_obs, :]
        Y = np.array(Y)

        # ML estimate of C and Q
        n_units = Y.shape[0]
        n_states = X.shape[0]
        C = np.zeros([n_units, n_states])
        pvalues = np.zeros([n_units, n_states])
        #C = np.zeros([n_states, n_units])
        glm_family = sm.families.Poisson()
        for k in range(n_units):
            model = sm.GLM(Y[k,:], X.T, family=glm_family)
            model_fit = model.fit()
            C[k,:] = model_fit.params
            pvalues[k,:] = model_fit.pvalues

        return C, pvalues

class PPFDecoder(bmi.BMI, bmi.Decoder):
    def __init__(self, filt, units, bounding_box, states, drives_neurons,
        states_to_bound, binlen=0.1, n_subbins=3, tslice=[-1,-1]):
        '''
        Initialize a PPF decoder

        Parameters
        ----------
        filt : PointProcessFilter instance (not named ppf to keep things generic)
            Generic point-process filter that does the actual observation decoding
        units : array-like
            N x 2 array of units, where each row is (chan, unit)
        bounding_box : tuple
            2-tuple of (lower bounds, upper bounds) for states that require 
            hard bounds (e.g. position bounds for a cursor to keep it from 
            going off the screen)
        states : list of strings
            List of variables known to the decoder
        drives_neurons : list of strings
            List of variables that the decoder uses to explain neural firing
        states_to_bound : list of strings
            List of variables to which to apply the hard bounds specified in 
            the 'bounding_box' argument
        binlen : float, optional, default = 0.1
            Bin-length specified in seconds. Gets rounded to a multiple of 1./60
            to match the update rate of the task
        n_subbins : int, optional, default = 3
            Neural observations are always acquired at the 60Hz screen update rate.
            This parameter explains how many bins to sub-divide the observations 
            into. Default of 3 is intended to correspond to ~180Hz / 5.5ms bins
        tslice : array_like, optional, default=[-1, -1]
            start and end times for the neural data used to train, e.g. from the .plx file
            No idea why this was specified as an attribute of the decoder instead of 
            the database, and changing it now is a high-risk, low-payoff option
        '''
        self.filt = filt
        self.filt._init_state()
        self.units = np.array(units, dtype=np.int32)
        self.binlen = binlen
        self.bounding_box = bounding_box
        self.states = states
        self.tslice = tslice
        self.states_to_bound = states_to_bound
        self.states = states
        self.drives_neurons = drives_neurons
        self.n_subbins = n_subbins

    def __call__(self, obs_t, **kwargs):
        '''
        '''
        # The PPF model predicts that at most one spike can be observed in 
        # each bin; if more are observed, squash the counts
        obs_t[obs_t > 1] = 1

        outputs = []
        for k in range(self.n_subbins):
            outputs.append(self.predict(obs_t[:,k], **kwargs))

        return np.vstack(outputs).T

    def get_filter(self):
        return self.filt 

    def predict(self, spike_counts, target=None, speed=0.05, assist_level=0., **kwargs):
        """
        Run decoder, assist, and bound any states
        """
        # TODO optimal feedback control assist
        if assist_level > 0 and target is not None:
            target_state = np.hstack([target, np.zeros(3), 1])
            target_state = np.mat(target_state).reshape(-1,1)
        else:
            target_state = None

        I = np.mat(np.eye(3))
        alpha = 1.622691378496069e-03 #6.458204410254785e-03
        gamma = 1.036424261334212e-03 #3.029680657880600e-03
        self.filt.F = assist_level*np.hstack([alpha*I, gamma*I, np.zeros([3,1])])

        # re-format as a 1D col vec
        spike_counts = np.mat(spike_counts.reshape(-1,1))

        # Run the filter
        self.filt(spike_counts, target_state=target_state)

        # Bound cursor, if any hard bounds for states are applied
        self.bound_state()

        state = self.filt.get_mean()
        return state
