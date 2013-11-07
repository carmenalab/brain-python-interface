'''Needs docs'''

import numpy as np

import bmi
from bmi import GaussianState
import statsmodels.api as sm # GLM fitting module
from scipy.io import loadmat
import time
import cmath
import feedback_controllers
import pickle

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
        self.spike_rate_dt = dt
        
        self.B = B
        self.F = F

        if is_stochastic == None:
            n_states = A.shape[0]
            self.is_stochastic = np.ones(n_states, dtype=bool)
        else:
            self.is_stochastic = np.array(is_stochastic)
        
        self.state_noise = GaussianState(0.0, W)
        self._pickle_init()

    def _pickle_init(self):
        """Code common to unpickling and initialization
        """
        nS = self.A.shape[0]
        offset_row = np.zeros(nS)
        offset_row[-1] = 1
        self.include_offset = np.array_equal(np.array(self.A)[-1, :], offset_row)

        self.spike_rate_dt = self.dt

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
        if np.any((lambda_predict * self.spike_rate_dt) > 1): 
            raise ValueError("Cell exploded!")

    def _obs_prob(self, state):
        Loglambda_predict = self.C * state.mean
        lambda_predict = np.exp(Loglambda_predict)/self.spike_rate_dt

        nan_inds = np.isnan(lambda_predict)
        lambda_predict[nan_inds] = 0

        # check max rate is less than 1 b/c it's a probability
        rate_too_high_inds = ((lambda_predict * self.spike_rate_dt) > 1)
        lambda_predict[rate_too_high_inds] = 1./self.spike_rate_dt

        # check min rate is > 0
        rate_too_low_inds = (lambda_predict < 0)
        lambda_predict[rate_too_low_inds] = 0
        #assert np.all(lambda_predict > 0)

        #self._check_valid(self, lambda_predict, id)
        invalid_inds = nan_inds | rate_too_high_inds | rate_too_low_inds
        if np.any(invalid_inds):
            pass
            #print np.nonzero(invalid_inds.ravel()[0])
        return lambda_predict
    
    def _ssm_pred(self, state, target_state=None):
        A = self.A
        B = self.B
        F = self.F
        if target_state == None:
            return A*state + self.state_noise
        else:
            return (A - B*F)*state + B*F*target_state + self.state_noise

    def _forward_infer(self, st, obs_t, target_state=None):
        obs_t = np.mat(obs_t.reshape(-1,1))
        C = self.C
        n_obs, n_states = C.shape
        
        dt = self.spike_rate_dt
        inds, = np.nonzero(self.is_stochastic)
        mesh = np.ix_(inds, inds)
        A = self.A
        W = self.W
        C = C[:,inds]

        x_prev, P_prev = st.mean, st.cov
        B = self.B
        F = self.F
        if target_state == None or np.any(np.isnan(target_state)):
            x_pred = A*x_prev
            P_pred = A*P_prev*A.T + W
        else:
            x_pred = A*x_prev + B*F*(target_state - x_prev)
            #import pdb; pdb.set_trace()
            #x_pred = A*x_prev - B*F*(x_prev - target_state)
            P_pred = (A-B*F) * P_prev * (A-B*F).T + W
        P_pred = P_pred[mesh]

        Loglambda_predict = self.C * x_pred 
        exp = np.vectorize(lambda x: np.real(cmath.exp(x)))
        lambda_predict = exp(np.array(Loglambda_predict).ravel())/dt

        Q_inv = np.mat(np.diag(lambda_predict*dt))

        if np.linalg.cond(P_pred) > 1e5:
            P_est = P_pred;
        else:
            P_est = (P_pred.I + C.T*np.mat(np.diag(lambda_predict*dt))*C).I

        # inflate P_est
        P_est_full = np.mat(np.zeros([n_states, n_states]))
        P_est_full[mesh] = P_est
        P_est = P_est_full 

        unpred_spikes = obs_t - np.mat(lambda_predict*dt).reshape(-1,1)

        x_est = np.mat(np.zeros([n_states, 1]))
        x_est = x_pred + P_est*self.C.T*unpred_spikes
        post_state = GaussianState(x_est, P_est)

        return post_state

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

    def tomlab(self, unit_scale=1.):
        '''
        convert the beta to the same format as MATLAB PPF decoders
        '''
        return np.array(np.hstack([self.C[:,-1], unit_scale*self.C[:,self.is_stochastic]])).T

    @classmethod
    def frommlab(self, beta_mat):
        return np.vstack([beta_mat[1:,:], beta_mat[0,:]]).T
        

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
        self.bmicount = 0
        self._pickle_init()

    def _pickle_init(self):
        ### # initialize the F_assist matrices
        ### # TODO this needs to be its own function...
        ### tau_scale = (28.*28)/1000/3. * np.array([18., 12., 6, 3., 2.5, 1.5])
        ### num_assist_levels = len(tau_scale)
        ### 
        ### w_x = 1;
        ### w_v = 3*tau_scale**2/2;
        ### w_r = 1e6*tau_scale**4;
        ### 
        ### I = np.eye(3)
        ### self.filt.B = np.bmat([[0*I], 
        ###               [self.filt.dt/1e-3 * I],
        ###               [np.zeros([1, 3])]])

        ### F = []
        ### F.append(np.zeros([3, 7]))
        ### for k in range(num_assist_levels):
        ###     Q = np.mat(np.diag([w_x, w_x, w_x, w_v[k], w_v[k], w_v[k], 0]))
        ###     R = np.mat(np.diag([w_r[k], w_r[k], w_r[k]]))

        ###     F_k = np.array(feedback_controllers.dlqr(self.filt.A, self.filt.B, Q, R, eps=1e-15))
        ###     F.append(F_k)
        ### 
        ### self.F_assist = np.dstack([np.array(x) for x in F]).transpose([2,0,1])
        #if not hasattr(self, 'F_assist'):
        self.F_assist = pickle.load(open('/storage/assist_params/assist_20levels.pkl'))
        self.n_assist_levels = len(self.F_assist)
        self.prev_assist_level = self.n_assist_levels

    def __call__(self, obs_t, **kwargs):
        '''
        '''
        # The PPF model predicts that at most one spike can be observed in 
        # each bin; if more are observed, squash the counts
        obs_t = obs_t.copy()
        obs_t[obs_t > 1] = 1

        # Infer the number of sub-bins from the size of the spike counts mat to decode
        n_subbins = obs_t.shape[1]

        outputs = []
        for k in range(n_subbins):
            outputs.append(self.predict(obs_t[:,k], **kwargs))

        return np.vstack(outputs).T

    def get_filter(self):
        return self.filt 

    def predict(self, spike_counts, target=None, speed=0.05, assist_level=0, **kwargs):
        """
        Run decoder, assist, and bound any states
        """
        #F_assist = self.F_assist #loadmat('/Users/sgowda/Desktop/ppf_code_1023/F_assist.mat')['F']

        # TODO optimal feedback control assist
        if target is not None:
            target_state = np.hstack([target, np.zeros(3), 1])
            target_state = np.mat(target_state).reshape(-1,1)
        else:
            target_state = None

        #I = np.mat(np.eye(3))
        #F = F_assist[:,:,int(assist_level)]
        #alpha = F[0,0]
        #gamma = F[0,2]
        #self.filt.F = np.mat( np.hstack([alpha*I, gamma*I, np.zeros([3,1])]) )
        assist_level_idx = min(int(assist_level * self.n_assist_levels), self.n_assist_levels-1)
        if assist_level_idx < self.prev_assist_level:
            print "assist_level_idx decreasing to", assist_level_idx
            self.prev_assist_level = assist_level_idx
        self.filt.F = np.mat(self.F_assist[assist_level_idx])
        #self.filt.B = np.mat( np.vstack([0*I, self.filt.dt*I*1000., np.zeros([1,3]) ]) )

        # re-format as a 1D col vec
        spike_counts = np.mat(spike_counts.reshape(-1,1))

        # Run the filter
        self.filt(spike_counts, target_state=target_state)

        # Bound cursor, if any hard bounds for states are applied
        self.bound_state()

        state = self.filt.get_mean()
        return state
