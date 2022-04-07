'''
Classes for BMI decoding using the Point-process filter. 
'''

import numpy as np

from . import bmi
from .bmi import GaussianState
import cmath

class PointProcessFilter(bmi.GaussianStateHMM):
    """
    Low-level Point-process filter, agnostic to application

    Model: 
       x_{t+1} = Ax_t + Bu_t + w_t; w_t ~ N(0, W)
       log(y_t) = Cx_t

    See Shanechi et al., "Feedback-Controlled Parallel Point Process Filter for 
    Estimation of Goal-Directed Movements From Neural Signals", IEEE TNSRE, 2013
    for mathematical details.
    """
    model_attrs = ['A', 'W', 'C']

    def __init__(self, A=None, W=None, C=None, dt=None, is_stochastic=None, B=0, F=0):
        '''
        Constructor for PointProcessFilter

        Parameters
        ----------
        A : np.mat
            Model of state transition matrix
        W : np.mat
            Model of process noise covariance
        C : np.mat
            Model of conditional distribution between observations and hidden state
            log(obs) = C * hidden_state
        dt : float
            Discrete-time sampling rate of the filter. Used to map spike counts to spike rates
        B : np.ndarray, optional
            Control input matrix
        F : np.ndarray, optional
            State-space feedback gain matrix to drive state back to equilibrium state.
        is_stochastic : np.array, optional
            Array of booleans specifying for each state whether it is stochastic. 
            If 'None' specified, all states are assumed to be stochastic

        Returns
        -------
        KalmanFilter instance
        '''
        if A is None and W is None and C is None and dt is None:
            ## This condition should only be true in the unpickling phase
            pass            
        else:
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
        """
        Code common to unpickling and initialization
        """
        nS = self.A.shape[0]
        offset_row = np.zeros(nS)
        offset_row[-1] = 1
        self.include_offset = np.array_equal(np.array(self.A)[-1, :], offset_row)

        self.spike_rate_dt = self.dt

        if not hasattr(self, 'B'): self.B = 0
        if not hasattr(self, 'F'): self.F = 0

    def init_noise_models(self):
        '''
        see bmi.GaussianStateHMM.init_noise_models for documentation
        '''
        self.state_noise = GaussianState(0.0, self.W)
        self.id = np.zeros([1, self.C.shape[0]])        

    def _check_valid(self, lambda_predict):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        if np.any((lambda_predict * self.spike_rate_dt) > 1): 
            raise ValueError("Cell exploded!")

    def _obs_prob(self, state):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''
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

        invalid_inds = nan_inds | rate_too_high_inds | rate_too_low_inds
        if np.any(invalid_inds):
            pass
            #print np.nonzero(invalid_inds.ravel()[0])
        return lambda_predict

    def _forward_infer(self, st, obs_t, Bu=None, u=None, x_target=None, F=None, obs_is_control_independent=False, **kwargs):
        '''
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------
        '''
        if np.any(obs_t > 1): 
            raise Exception
        using_control_input = (Bu is not None) or (u is not None) or (x_target is not None)
        if x_target is not None:
            x_target = np.mat(x_target[:,0].reshape(-1,1))
        target_state = x_target

        obs_t = np.mat(obs_t.reshape(-1,1))
        C = self.C
        n_obs, n_states = C.shape
        
        dt = self.spike_rate_dt
        inds, = np.nonzero(self.is_stochastic)
        mesh = np.ix_(inds, inds)
        A = self.A
        W = self.W
        C = C[:,inds]


        # print np.array(x_target).ravel()
        pred_state = self._ssm_pred(st, target_state=x_target, Bu=Bu, u=u, F=F)
        x_pred, P_pred = pred_state.mean, pred_state.cov
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
        self.neural_push = P_est*self.C.T*obs_t
        self.P_est = P_est
        post_state = GaussianState(x_est, P_est)
        return post_state

    def __getstate__(self):
        '''
        Return model parameters to be pickled. Overrides the default __getstate__ so that things like the P matrix aren't pickled.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        dict
        '''
        return dict(A=self.A, W=self.W, C=self.C, dt=self.dt, B=self.B, 
                    is_stochastic=self.is_stochastic)

    def tomlab(self, unit_scale=1.):
        '''
        Convert to the MATLAB beta matrix convention from the one used here (different state order, transposed)
        '''
        return np.array(np.hstack([self.C[:,-1], unit_scale*self.C[:,self.is_stochastic]])).T

    @classmethod
    def frommlab(self, beta_mat):
        '''
        Convert from the MATLAB beta matrix convention to the one used here (different state order, transposed)
        '''
        return np.vstack([beta_mat[1:,:], beta_mat[0,:]]).T

    @classmethod
    def MLE_obs_model(cls, hidden_state, obs, include_offset=True, drives_obs=None):
        """
        Unconstrained ML estimator of {C, } given observations and
        the corresponding hidden states
        Docstring    
        
        Parameters
        ----------
        
        Returns
        -------            
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
        import statsmodels.api as sm
        glm_family = sm.families.Poisson()
        for k in range(n_units):
            model = sm.GLM(Y[k,:], X.T, family=glm_family)
            try:
                model_fit = model.fit()
                C[k,:] = model_fit.params
                pvalues[k,:] = model_fit.pvalues                
            except:
                pvalues[k,:] = np.nan

        return C, pvalues



class OneStepMPCPointProcessFilter(PointProcessFilter):
    '''
    Use MPC with a horizon of 1 to predict 
    '''
    attrs_to_pickle = ['A', 'W', 'C']
    def _pickle_init(self):
        super(OneStepMPCPointProcessFilter, self)._pickle_init()

        self.prev_obs = None
        if not hasattr(self, 'mpc_cost_step'):
            mpc_cost_half_life = 1200.
            batch_time = 0.1
            self.mpc_cost_step = np.exp(np.log(0.5) / (mpc_cost_half_life/batch_time))
            self.ESS = 1000

    def _ssm_pred(self, state, u=None, Bu=None, target_state=None, F=None):
        ''' Docstring
        Run the "predict" step of the Kalman filter/HMM inference algorithm:
            x_{t+1|t} = N(Ax_{t|t}, AP_{t|t}A.T + W)

        Parameters
        ----------
        state: GaussianState instance
            State estimate and estimator covariance of current state
        u: np.mat 
        

        Returns
        -------
        GaussianState instance
            Represents the mean and estimator covariance of the new state estimate
        '''
        A = self.A

        dt = self.dt

        Loglambda_predict = self.C * state.mean 
        exp = np.vectorize(lambda x: np.real(cmath.exp(x)))
        lambda_predict = exp(np.array(Loglambda_predict).ravel())/dt

        Q_inv = np.mat(np.diag(lambda_predict*dt))

        if self.prev_obs is not None:
            y_ref = self.prev_obs
            G = self.C.T * Q_inv
            D = G * self.C
            D[:,-1] = 0
            D[-1,:] = 0

            # Solve for R
            R = 200*D
            
            alpha = A*state
            v = np.linalg.pinv(R + D)*(G*y_ref - D*alpha.mean)
        else:
            v = np.zeros_like(state.mean)

        if Bu is not None:
            return A*state + Bu + self.state_noise + v
        elif u is not None:
            Bu = self.B * u
            return A*state + Bu + self.state_noise + v
        elif target_state is not None:
            B = self.B
            F = self.F
            return (A - B*F)*state + B*F*target_state + self.state_noise  + v
        else:
            return A*state + self.state_noise + v

    def _forward_infer(self, st, obs_t, **kwargs):
        res = super(OneStepMPCPointProcessFilter, self)._forward_infer(st, obs_t, **kwargs)

        # if not (self.prev_obs is None):
        #     # Update the sufficient statistics for the R matrix
        #     l = self.mpc_cost_step
        #     self.E00 = l*self.E00 + (1-l)*(self.prev_obs - self.C*self.A*st.mean)*(self.prev_obs - self.C*self.A*st.mean).T
        #     self.E01 = l*self.E01 + (1-l)*(self.prev_obs - self.C*self.A*st.mean)*(obs_t - self.C*self.A*st.mean).T

        self.prev_obs = obs_t
        return res    

class OneStepMPCPointProcessFilterCovFb(OneStepMPCPointProcessFilter):
    def _ssm_pred(self, state, **kwargs):
        ''' Docstring
        Run the "predict" step of the Kalman filter/HMM inference algorithm:
            x_{t+1|t} = N(Ax_{t|t}, AP_{t|t}A.T + W)

        Parameters
        ----------
        state: GaussianState instance
            State estimate and estimator covariance of current state
        u: np.mat 
        

        Returns
        -------
        GaussianState instance
            Represents the mean and estimator covariance of the new state estimate
        '''
        A = self.A

        dt = self.dt

        Loglambda_predict = self.C * state.mean 
        exp = np.vectorize(lambda x: np.real(cmath.exp(x)))
        lambda_predict = exp(np.array(Loglambda_predict).ravel())/dt

        Q_inv = np.mat(np.diag(lambda_predict*dt))

        from .bmi import GaussianState
        if (self.prev_obs is not None) and (self.r_scale < np.inf):
            y_ref = self.prev_obs
            G = self.C.T * Q_inv
            D = G * self.C
            D[:,-1] = 0
            D[-1,:] = 0

            # Solve for R
            R = self.r_scale*D
            
            alpha = A*state
            v = np.linalg.pinv(R + D)*(G*y_ref - D*alpha.mean)
            I = np.mat(np.eye(D.shape[0]))
            C = self.C
            A = (I - G*C) * self.A
            mean = A*state.mean + G*y_ref
            cov = A*state.cov*A.T + self.W

            return GaussianState(mean, cov)
        else:
            return A*state + self.state_noise
            # v = np.zeros_like(state.mean)



class PPFDecoder(bmi.BMI, bmi.Decoder):
    def __call__(self, obs_t, **kwargs):
        '''
        see bmi.Decoder.__call__ for docs
        '''
        # The PPF model predicts that at most one spike can be observed in 
        # each bin; if more are observed, squash the counts
        # (make a copy of the observation matrix prior to squashing)
        obs_t = obs_t.copy()
        obs_t[obs_t > 1] = 1
        return super(PPFDecoder, self).__call__(obs_t, **kwargs)

    def shuffle(self):
        '''
        Shuffle the neural model
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        import random
        inds = list(range(self.filt.C.shape[0]))
        random.shuffle(inds)

        # shuffle rows of C
        self.filt.C = self.filt.C[inds, :]

    def compute_suff_stats(self, hidden_state, obs, include_offset=True):
        '''
        Calculate initial estimates of the parameter sufficient statistics used in the RML update rules

        Parameters
        ----------
        hidden_state : np.ndarray of shape (n_states, n_samples)
            Examples of the hidden state x_t taken from training seed data.  
        obs : np.ndarray of shape (n_features, n_samples)
            Multiple neural observations paired with each of the hidden state examples
        include_offset : bool, optional
            If true, a state of all 1's is added to the hidden_state to represent mean offsets. True by default

        Returns
        -------
        R : np.ndarray of shape (n_states, n_states)
            Proportional to covariance of the hidden state samples 
        S : np.ndarray of shape (n_features, n_states)
            Proportional to cross-covariance between 
        T : np.ndarray of shape (n_features, n_features)
            Proportional to covariance of the neural observations
        ESS : float
            Effective number of samples. In the initialization, this is just the 
            dimension of the array passed in, but the parameter can become non-integer 
            during the update procedure as old parameters are "forgotten".
        '''
        n_obs = obs.shape[0]
        nS = hidden_state.shape[0]

        H = np.zeros([n_obs, nS, nS])
        M = np.zeros([n_obs, nS])
        S = np.zeros([n_obs, nS])

        C = self.filt.C[:,self.drives_neurons]

        X = np.array(hidden_state)
        T = X.shape[1]
        if include_offset:
            if not np.all(X[-1,:] == 1):
                X = np.vstack([ X, np.ones([1,T]) ])

        for k in range(n_obs):
            Mu = np.exp(np.dot(C[k,:], X)).ravel()
            Y = obs[k,:]
            H[k] = np.dot((np.tile(np.array(-Mu), [nS, 1]) * X), X.T)
            M[k] = np.dot(Mu, X.T)
            S[k] = np.dot(Y, X.T)

        self.H = H
        self.S = S
        self.M = M

    @property 
    def n_features(self):
        return self.filt.C.shape[0]
