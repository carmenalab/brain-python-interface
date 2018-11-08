'''
Classes for BMI decoding using the Kalman filter. 
'''

import numpy as np
from scipy.io import loadmat

from . import bmi
import pickle
import re

class KalmanFilter(bmi.GaussianStateHMM):
    """
    Low-level KF, agnostic to application

    Model: 
       x_{t+1} = Ax_t + w_t;   w_t ~ N(0, W)
           y_t = Cx_t + q_t;   q_t ~ N(0, Q)
    """
    model_attrs = ['A', 'W', 'C', 'Q', 'C_xpose_Q_inv', 'C_xpose_Q_inv_C']
    attrs_to_pickle = ['A', 'W', 'C', 'Q', 'C_xpose_Q_inv', 'C_xpose_Q_inv_C', 'R', 'S', 'T', 'ESS']

    def __init__(self, A=None, W=None, C=None, Q=None, is_stochastic=None):
        '''
        Constructor for KalmanFilter    

        Parameters
        ----------
        A : np.mat, optional
            Model of state transition matrix
        W : np.mat, optional
            Model of process noise covariance
        C : np.mat, optional
            Model of conditional distribution between observations and hidden state
        Q : np.mat, optional
            Model of observation noise covariance
        is_stochastic : np.array, optional
            Array of booleans specifying for each state whether it is stochastic. 
            If 'None' specified, all states are assumed to be stochastic

        Returns
        -------
        KalmanFilter instance
        '''
        if A is None and W is None and C is None and Q is None:
            ## This condition should only be true in the unpickling phase
            pass
        else:
            self.A = np.mat(A)
            self.W = np.mat(W)
            self.C = np.mat(C)
            self.Q = np.mat(Q)

            if is_stochastic is None:
                n_states = self.A.shape[0]
                self.is_stochastic = np.ones(n_states, dtype=bool)
            else:
                self.is_stochastic = is_stochastic
            
            self.state_noise = bmi.GaussianState(0.0, self.W)
            self.obs_noise = bmi.GaussianState(0.0, self.Q)
            self._pickle_init()

    def _pickle_init(self):
        """Code common to unpickling and initialization
        """
        nS = self.A.shape[0]
        offset_row = np.zeros(nS)
        offset_row[-1] = 1
        self.include_offset = np.array_equal(np.array(self.A)[-1, :], offset_row)

        self.alt = nS < self.C.shape[0] # No. of states less than no. of observations
        attrs = list(self.__dict__.keys())
        if not 'C_xpose_Q_inv_C' in attrs:
            C, Q = self.C, self.Q 
            self.C_xpose_Q_inv = C.T * np.linalg.pinv(Q)
            self.C_xpose_Q_inv_C = C.T * np.linalg.pinv(Q) * C

        try:
            self.is_stochastic
        except:
            n_states = self.A.shape[0]
            self.is_stochastic = np.ones(n_states, dtype=bool)

    def _obs_prob(self, state):
        '''
        Predict the observations based on the model parameters:
            y_est = C*x_t + Q

        Parameters
        ----------
        state : bmi.GaussianState instance
            The model-predicted state

        Returns
        -------
        bmi.GaussianState instance
            the model-predicted observations
        '''
        return self.C * state + self.obs_noise

    def _forward_infer(self, st, obs_t, Bu=None, u=None, x_target=None, F=None, obs_is_control_independent=True, **kwargs):
        '''
        Estimate p(x_t | ..., y_{t-1}, y_t)

        Parameters
        ----------
        st : GaussianState
            Current estimate (mean and cov) of hidden state
        obs_t : np.mat of shape (N, 1)
             ARG_DESCR
        Bu : DATA_TYPE, optional, default=None
             ARG_DESCR
        u : DATA_TYPE, optional, default=None
             ARG_DESCR
        x_target : DATA_TYPE, optional, default=None
             ARG_DESCR
        obs_is_control_independent : bool, optional, default=True
             ARG_DESCR
        kwargs : optional kwargs
            ARG_DESCR

        Returns
        -------
        GaussianState
            New state estimate incorporating the most recent observation

        '''
        using_control_input = (Bu is not None) or (u is not None) or (x_target is not None)
        pred_state = self._ssm_pred(st, target_state=x_target, Bu=Bu, u=u, F=F)

        C, Q = self.C, self.Q
        P = pred_state.cov

        K = self._calc_kalman_gain(P)
        I = np.mat(np.eye(self.C.shape[1]))
        D = self.C_xpose_Q_inv_C
        KC = P*(I - D*P*(I + D*P).I)*D
        F = (I - KC)*self.A

        post_state = pred_state

        if obs_is_control_independent and using_control_input:
            post_state.mean += -KC*self.A*st.mean + K*obs_t
        else:
            post_state.mean += -KC*pred_state.mean + K*obs_t

        post_state.cov = (I - KC) * P 

        return post_state

    def set_state_cov(self, n_steps):
        C, Q = self.C, self.Q
        A, W = self.A, self.W
        P = self.state.cov
        for k in range(n_steps):
            
            P = A*P*A.T + W

            K = self._calc_kalman_gain(P)
            I = np.mat(np.eye(self.C.shape[1]))
            D = self.C_xpose_Q_inv_C
            KC = P*(I - D*P*(I + D*P).I)*D
            P = (I - KC) * P 

        return P

    def _calc_kalman_gain(self, P):
        '''
        Calculate Kalman gain using the 'alternate' definition

        Parameters
        ----------
        P : np.matrix
            Prediciton covariance matrix, i.e., cov(x_{t+1} | y_1, \cdots, y_t)

        Returns
        -------
        K : np.matrix
            Kalman gain matrix for the input next state prediciton covariance.        
        '''
        nX = P.shape[0]
        I = np.mat(np.eye(nX))
        D = self.C_xpose_Q_inv_C
        L = self.C_xpose_Q_inv
        K = P * (I - D*P*(I + D*P).I) * L
        return K

    def get_sskf(self, tol=1e-15, return_P=False, dtype=np.array, max_iter=4000,
        verbose=False, return_Khist=False, alt=True):
        """Calculate the steady-state KF matrices

        value of P returned is the posterior error cov, i.e. P_{t|t}

        Parameters
        ----------

        Returns
        -------        
        """ 
        A, W, C, Q = np.mat(self.A), np.mat(self.W), np.mat(self.C), np.mat(self.Q)

        nS = A.shape[0]
        P = np.mat(np.zeros([nS, nS]))
        I = np.mat(np.eye(nS))

        D = self.C_xpose_Q_inv_C 

        last_K = np.mat(np.ones(C.T.shape))*np.inf
        K = np.mat(np.ones(C.T.shape))*0

        K_hist = []

        iter_idx = 0
        last_P = None
        while np.linalg.norm(K-last_K) > tol and iter_idx < max_iter:
            P = A*P*A.T + W 
            last_K = K
            K = self._calc_kalman_gain(P)
            K_hist.append(K)
            KC = P*(I - D*P*(I + D*P).I)*D
            last_P = P
            P -= KC*P;
            iter_idx += 1
        if verbose: 
            print(("Converged in %d iterations--error: %g" % (iter_idx, np.linalg.norm(K-last_K)))) 
    
        n_state_vars, n_state_vars = A.shape
        F = (np.mat(np.eye(n_state_vars, n_state_vars)) - KC) * A
    
        if return_P and return_Khist:
            return dtype(F), dtype(K), dtype(last_P), K_hist
        elif return_P:
            return dtype(F), dtype(K), dtype(last_P)
        elif return_Khist:
            return dtype(F), dtype(K), K_hist
        else:
            return dtype(F), dtype(K)


    def get_kalman_gain_seq(self, N=1000, tol=1e-10, verbose=False):
        '''
        Calculate K_t for times {0, 1, ..., N}

        Parameters
        ----------
        N : int, optional
            Number of steps to calculate Kalman gain for, default = 1000
        tol : float, optional
            Tolerance on K matrix convergence, default = 1e-10
        verbose : bool, optional
            Print intermediate/debugging information if true, default=False

        Returns
        -------
        list
            [K_0, K_1, ..., K_{N-1}]
        '''
        A, W, H, Q = np.mat(self.kf.A), np.mat(self.kf.W), np.mat(self.kf.H), np.mat(self.kf.Q)
        P = np.mat( np.zeros(A.shape) )
        K = [None]*N
        
        ss_idx = None # index at which K is steady-state (within tol)
        for n in range(N):
            if not ss_idx == None and n > ss_idx:
                K[n] = K[ss_idx]
            else:
                P = A*P*A.T + W 
                K[n] = (P*H.T)*linalg.pinv(H*P*H.T + Q);
                P -= K[n]*H*P;
                if n > 0 and np.linalg.norm(K[n] - K[n-1]) < tol: 
                    ss_idx = n
                    if verbose: 
                        print(("breaking after %d iterations" % n))

        return K, ss_idx

    def get_kf_system_mats(self, T):
        """
        KF system matrices

        x_{t+1} = F_t*x_t + K_t*y_t 

        Parameters
        ----------
        T : int 
            Number of system iterations to calculate (F_t, K_t)

        Returns
        -------
        tuple of lists
            Each element of the tuple is (F_t, K_t) for a given 't'

        """
        F = [None]*T
        K, ss_idx = self.get_kalman_gain_seq(N=T, verbose=False)
        nX = self.kf.A.shape[0]
        I = np.mat(np.eye(nX))
        
        for t in range(T):
            if t > ss_idx: F[t] = F[ss_idx]
            else: F[t] = (I - K[t]*self.kf.H)*self.kf.A
        
        return F, K

    @classmethod
    def MLE_obs_model(self, hidden_state, obs, include_offset=True, drives_obs=None,
        regularizer=None):
        """
        Unconstrained ML estimator of {C, Q} given observations and
        the corresponding hidden states

        Parameters
        ----------
        include_offset : bool, optional, default=True
            A row of all 1's is added as the last row of hidden_state if one is not already present

        Returns
        -------        
        """
        assert hidden_state.shape[1] == obs.shape[1], "different numbers of time samples: %s vs %s" % (str(hidden_state.shape), str(obs.shape))
    
        if isinstance(hidden_state, np.ma.core.MaskedArray):
            mask = ~hidden_state.mask[0,:] # NOTE THE INVERTER 
            inds = np.nonzero([ mask[k]*mask[k+1] for k in range(len(mask)-1)])[0]
    
            X = np.mat(hidden_state[:,mask])
            T = len(np.nonzero(mask)[0])
    
            Y = np.mat(obs[:,mask])
            if include_offset:
                if not np.all(X[-1,:] == 1):
                    X = np.vstack([ X, np.ones([1,T]) ])
        else:
            num_hidden_state, T = hidden_state.shape
            X = np.mat(hidden_state)
            if include_offset:
                if not np.all(X[-1,:] == 1):
                    X = np.vstack([ X, np.ones([1,T]) ])
            Y = np.mat(obs)
    
        n_states = X.shape[0]
        if not drives_obs is None:
            X = X[drives_obs, :]
            
        # ML estimate of C and Q
        if regularizer is None:
            C = np.mat(np.linalg.lstsq(X.T, Y.T)[0].T)
        else:
            x = X.T
            y = Y.T
            XtX_lamb = x.T.dot(x) + regularizer * np.eye(x.shape[1])
            XtY = x.T.dot(y)
            C = np.linalg.solve(XtX_lamb, XtY).T
        Q = np.cov(Y - C*X, bias=1)

        if np.ndim(Q) == 0:
            # if "obs" only has 1 feature, Q might get collapsed to a scalar
            Q = np.mat(Q.reshape(1,1))

        if not drives_obs is None:
            n_obs = C.shape[0]
            C_tmp = np.zeros([n_obs, n_states])
            C_tmp[:,drives_obs] = C
            C = C_tmp
        return (C, Q)
    
    @classmethod 
    def MLE_state_space_model(self, hidden_state, include_offset=True):
        '''
        Train state space model for KF from fully observed hidden state

        Parameters
        ----------
        hidden_state : np.ndarray of shape (N, T)
            N = dimensionality of state vector, T = number of observations
        include_offset : boolean, optional, default=False
            if True, append a "1" to each state vector to add an offset term into the 
            regression

        Returns
        -------        
        A : np.ndarray of shape (N, N)
        W : np.ndarray of shape (N, N)
        '''
        X = hidden_state
        T = hidden_state.shape[1]
        if include_offset:
            X = np.vstack([ X, np.ones([1,T]) ])        
        X1 = X[:,:-1]
        X2 = X[:,1:]
        A = np.linalg.lstsq(X1.T, X2.T)[0].T
        W = np.cov(X2 - np.dot(A, X1), bias=1)
        return A, W

    def set_steady_state_pred_cov(self):
        '''
        Calculate the steady-state prediction covariance and set the current state prediction covariance to the steady-state value
        '''

        A, W, C, Q = np.mat(self.A), np.mat(self.W), np.mat(self.C), np.mat(self.Q)
        D = self.C_xpose_Q_inv_C 
        nS = A.shape[0]
        P = np.mat(np.zeros([nS, nS]))
        I = np.mat(np.eye(nS))

        last_K = np.mat(np.ones(C.T.shape))*np.inf
        K = np.mat(np.ones(C.T.shape))*0

        iter_idx = 0
        for iter_idx in range(40):
            P = A*P*A.T + W
            last_K = K
            KC = P*(I - D*P*(I + D*P).I)*D
            P -= KC*P;

        # TODO fix
        P[0:3, 0:3] = 0
        F, K = self.get_sskf()
        F = (I - KC)*A
        self._init_state(init_state=self.state.mean, init_cov=P)

    def get_K_null(self):
        '''
        $$y_{null} = K_{null} * y_t$$ gives the "null" component of the spike inputs, i.e. $$K_t*y_{null} = 0_{N\times 1}$$
        Parameters
        ----------

        Returns
        -------        
        '''
        F, K = self.get_sskf()
        K = np.mat(K)
        n_neurons = K.shape[1]
        K_null = np.eye(n_neurons) - np.linalg.pinv(K) * K
        return K_null

class KalmanFilterDriftCorrection(KalmanFilter):
    attrs_to_pickle = ['A', 'W', 'C', 'Q', 'C_xpose_Q_inv',
        'C_xpose_Q_inv_C', 'R', 'S', 'T', 'ESS', 'drift_corr','prev_drift_corr']
    noise_threshold = 96.*3.5

    def _init_state(self):
        if hasattr(self, 'prev_drift_corr'):
            self.drift_corr = self.prev_drift_corr.copy()
            print(('prev drift corr', np.mean(self.prev_drift_corr)))
        else:
            self.drift_corr = np.mat(np.zeros(( self.A.shape[0], 1)))
            self.prev_drift_corr = np.mat(np.zeros(( self.A.shape[0], 1)))

        if hasattr(self, 'noise_rej'):
            if self.noise_rej:
                print(('noise rej thresh: ', self.noise_rej_cutoff))
        else:
            self.noise_rej = False
        self.noise_cnt = 0

        super(KalmanFilterDriftCorrection, self)._init_state()
        
    def _forward_infer(self, st, obs_t, Bu=None, u=None, x_target=None, F=None, obs_is_control_independent=True, **kwargs):
        
        if self.noise_rej:
            if np.sum(obs_t) > self.noise_rej_cutoff:
                #print np.sum(obs_t), 'rejecting noise!'
                self.noise_cnt += 1 
                obs_t = np.mat(self.noise_rej_mFR).T
                

        state = super(KalmanFilterDriftCorrection, self)._forward_infer(st, obs_t, Bu=None, u=None, x_target=None, F=None, 
            obs_is_control_independent=True, **kwargs)

        ### Apply Drift Correction ###
        decoded_vel = state.mean.copy()
        state.mean[self.vel_ix] = decoded_vel[self.vel_ix] - self.drift_corr[self.vel_ix]

        ### Update Drift Correcton ###
        self.drift_corr[self.vel_ix] = self.drift_corr[self.vel_ix]*self.drift_rho + decoded_vel[self.vel_ix]*float(1. - self.drift_rho)
        self.prev_drift_corr = self.drift_corr.copy()

        return state

class PCAKalmanFilter(KalmanFilter):
    '''
    A modified KalmanFilter where the Kalman gain is confined to produce outputs in a lower-dimensional linear subspace, i.e. some principal component space
    '''
    def _forward_infer(self, st, obs_t, Bu=None, u=None, target_state=None, obs_is_control_independent=True, **kwargs):
        '''
        See KalmanFilter._forward_infer for docs
        '''
        using_control_input = (Bu is not None) or (u is not None) or (target_state is not None)
        pred_state = self._ssm_pred(st, target_state=target_state, Bu=Bu, u=u)

        C, Q = self.C, self.Q
        P = pred_state.cov

        try:
            M = self.M
            pca_offset = self.pca_offset
        except:
            print("couldn't extract PCA parameters!")
            M = 1
            pca_offset = 0

        K = self._calc_kalman_gain(P)
        I = np.mat(np.eye(self.C.shape[1]))
        D = self.C_xpose_Q_inv_C

        KC = K*C
        F = (I - KC)*self.A

        post_state = pred_state
        if obs_is_control_independent and using_control_input:
            post_state.mean += -KC*self.A*st.mean + M*K*obs_t + pca_offset
        else:
            post_state.mean += -KC*pred_state.mean + M*K*obs_t + pca_offset

        post_state.cov = (I - KC) * P 

        return post_state

    def __getstate__(self):
        '''
        See KalmanFilter.__getstate__ for docs
        '''
        data = super(PCAKalmanFilter, self).__getstate__()
        data['M'] = self.M
        data['pca_offset'] = self.pca_offset
        return data

    def __setstate__(self, state):
        '''
        See KalmanFilter.__setstate__ for docs
        '''
        super(PCAKalmanFilter, self).__setstate__(state)
        self.M = state['M']
        self.pca_offset = state['pca_offset']        

class FAKalmanFilter(KalmanFilter):

    def _forward_infer(self, st, obs_t, Bu=None, u=None, target_state=None, obs_is_control_independent=True, **kwargs):
        input_dict = {}
        if hasattr(self, 'FA_kwargs'):

            input_type = self.FA_input + '_input'
        
            input_dict['all_input'] = obs_t.copy()

            dmn = obs_t - self.FA_kwargs['fa_mu']
            shar = (self.FA_kwargs['fa_sharL'] * dmn)
            priv = (dmn - shar)
            main_shar = (self.FA_kwargs['fa_main_shared'] * dmn)
            main_priv = (dmn - main_shar)

            FA = self.FA_kwargs['FA_model']

            inp = obs_t.copy()
            if inp.shape[1] == 1:
                inp = inp.T # want 1 x neurons
            z = FA.transform(dmn.T)
            z = z.T #Transform to fact x 1
            z = z[:self.FA_kwargs['fa_main_shar_n_dim'], :] #only use number in main space

            input_dict['private_input'] = priv + self.FA_kwargs['fa_mu']
            input_dict['shared_input'] = shar + self.FA_kwargs['fa_mu']

            input_dict['private_scaled_input'] = np.multiply(priv, self.FA_kwargs['fa_priv_var_sc']) + self.FA_kwargs['fa_mu']
            input_dict['shared_scaled_input'] = np.multiply(shar, self.FA_kwargs['fa_shar_var_sc']) + self.FA_kwargs['fa_mu']

            input_dict['all_scaled_by_shar_input'] = np.multiply(dmn, self.FA_kwargs['fa_shar_var_sc']) + self.FA_kwargs['fa_mu']

            input_dict['sc_shared+unsc_priv_input'] = input_dict['shared_scaled_input'] + input_dict['private_input'] - self.FA_kwargs['fa_mu']
            input_dict['sc_shared+sc_priv_input'] = input_dict['shared_scaled_input'] + input_dict['private_scaled_input']- self.FA_kwargs['fa_mu']

            input_dict['main_shared_input'] = main_shar + self.FA_kwargs['fa_mu']
            input_dict['main_sc_shared_input'] = np.multiply(main_shar, self.FA_kwargs['fa_main_shared_sc']) + self.FA_kwargs['fa_mu']

            input_dict['main_sc_shar+unsc_priv_input'] = input_dict['main_sc_shared_input'] + input_dict['private_input'] - self.FA_kwargs['fa_mu']
            input_dict['main_sc_shar+sc_priv_input'] = input_dict['main_sc_shared_input'] + input_dict['private_scaled_input'] - self.FA_kwargs['fa_mu']
            input_dict['main_sc_private_input'] = np.multiply(main_priv, self.FA_kwargs['fa_main_private_sc']) + self.FA_kwargs['fa_mu']

            #z = self.FA_kwargs['u_svd'].T*self.FA_kwargs['uut_psi_inv']*dmn
            input_dict['split_input'] = np.vstack((z, main_priv))
            #print input_dict['split_input'].shape
            
            own_pc_trans = np.mat(self.FA_kwargs['own_pc_trans'])*np.mat(dmn)
            input_dict['pca_input'] = own_pc_trans + self.FA_kwargs['fa_mu']

            if input_type in list(input_dict.keys()):
                #print input_type
                obs_t_mod = input_dict[input_type]
            else: 
                print(input_type)
                raise Exception("Error in FA_KF input_type, none of the expected inputs")
        else:
            obs_t_mod = obs_t.copy()

        input_dict['task_input'] = obs_t_mod.copy()


        post_state = super(FAKalmanFilter, self)._forward_infer(st, obs_t_mod, Bu=Bu, u=u, target_state=target_state, 
            obs_is_control_independent=obs_is_control_independent, **kwargs)

        self.FA_input_dict = input_dict

        return post_state

class KFDecoder(bmi.BMI, bmi.Decoder):
    '''
    Wrapper for KalmanFilter specifically for the application of BMI decoding.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for KFDecoder   
        
        Parameters
        ----------
        *args, **kwargs : see riglib.bmi.bmi.Decoder for arguments
        
        Returns
        -------
        KFDecoder instance
        '''

        super(KFDecoder, self).__init__(*args, **kwargs)
        mFR = kwargs.pop('mFR', 0.)
        sdFR = kwargs.pop('sdFR', 1.)
        self.mFR = mFR
        self.sdFR = sdFR
        self.zeromeanunits = None
        self.zscore = False
        self.kf = self.filt

    def _pickle_init(self):
        super(KFDecoder, self)._pickle_init()
        if not hasattr(self.filt, 'B'):
            self.filt.B = np.mat(np.vstack([np.zeros([3,3]), np.eye(3)*1000*self.binlen, np.zeros(3)]))

        if not hasattr(self.filt, 'F'):
            self.filt.F = np.mat(np.zeros([3,7]))

    def init_zscore(self, mFR_curr, sdFR_curr):
        '''
        Initialize parameters for zcoring observations, if that feature is enabled in the decoder object
        
        Parameters
        ----------
        mFR_curr : np.array of shape (N,)
            Current mean estimates (as opposed to potentially old estimates already stored in the decoder)
        sdFR_curr : np.array of shape (N,)
            Current standard deviation estimates (as opposed to potentially old estimates already stored in the decoder)
        
        Returns
        -------
        None
        '''

        # if interfacing with Kinarm system, may mean and sd will be shape (n, 1)
        self.zeromeanunits, = np.nonzero(mFR_curr == 0) #find any units with a mean FR of zero for this session
        sdFR_curr[self.zeromeanunits] = np.nan # set mean and SD of quiet units to nan to avoid divide by 0 error
        mFR_curr[self.zeromeanunits] = np.nan
        #self.sdFR_ratio = self.sdFR/sdFR_curr
        #self.mFR_diff = mFR_curr-self.mFR
        #self.mFR_curr = mFR_curr
        self.mFR = mFR_curr
        self.sdFR = sdFR_curr
        self.zscore = True

    def update_params(self, new_params, steady_state=True):
        '''
        Update the decoder parameters if new parameters are available (e.g., by CLDA). See Decoder.update_params
        '''
        super(KFDecoder, self).update_params(new_params)

        # set the KF to the new steady state
        if steady_state:
            self.kf.set_steady_state_pred_cov()

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
        if 'kf' in state and 'filt' not in state:
            state['filt'] = state['kf']

        super(KFDecoder, self).__setstate__(state)

    def plot_K(self, **kwargs):
        '''
        Plot the Kalman gain weights
        
        Parameters
        ----------
        **kwargs : optional kwargs
            These are passed to the plot function (e.g., which rows to plot)
        
        Returns
        -------
        None
        '''

        F, K = self.kf.get_sskf()
        self.plot_pds(K.T, **kwargs)

    def shuffle(self, shuffle_baselines=False):
        '''
        Shuffle the neural model
        
        Parameters
        ----------
        shuffle_baselines : bool, optional, default = False
            If true, shuffle the estimates of the baseline firing rates in addition to the state-dependent neural tuning parameters.
        
        Returns
        -------
        None (shuffling is done on the current decoder object)        

        '''
        # generate random permutation
        import random
        inds = list(range(self.filt.C.shape[0]))
        random.shuffle(inds)

        # shuffle rows of C, and rows+cols of Q
        C_orig = self.filt.C.copy()
        self.filt.C = self.filt.C[inds, :]
        if not shuffle_baselines:
            self.filt.C[:,-1] = C_orig[:,-1]
        self.filt.Q = self.filt.Q[inds, :]
        self.filt.Q = self.filt.Q[:, inds]

        self.filt.C_xpose_Q_inv = self.filt.C.T * np.linalg.pinv(self.filt.Q.I)

        # RML sufficient statistics (S and T, but not R and ESS)
        # shuffle rows of S, and rows+cols of T
        try:
            self.filt.S = self.filt.S[inds, :]
            self.filt.T = self.filt.T[inds, :]
            self.filt.T = self.filt.T[:, inds]
        except AttributeError:
            # if this decoder never had the RML sufficient statistics
            #   (R, S, T, and ESS) as attributes of self.filt
            pass

    def change_binlen(self, new_binlen, screen_update_rate=60.0):
        '''
        Function to change the binlen of the KFDecoder analytically. 

        Parameters
        ----------
        new_binlen : float
            New bin length of the decoder, in seconds
        screen_update_rate: float, optional, default = 60Hz
            Rate at which the __call__ function will be called
        '''
        bin_gain = new_binlen / self.binlen
        self.binlen = new_binlen

        # Alter bminum, bmicount, # of subbins
        screen_update_period = 1./screen_update_rate
        if self.binlen < screen_update_period:
            self.n_subbins = int(screen_update_period / self.binlen)
            self.bmicount = 0
            if hasattr(self, 'bminum'):
                del self.bminum
        else:
            self.n_subbins = 1
            self.bminum = int(self.binlen / screen_update_period)
            self.bmicount = 0

        # change C matrix
        self.filt.C *= bin_gain
        self.filt.Q *= bin_gain**2
        self.filt.C_xpose_Q_inv *= 1./bin_gain

        # change state space Model
        # TODO generalize this beyond endpoint
        from . import state_space_models
        A, W = self.ssm.get_ssm_matrices(update_rate=new_binlen)
        self.filt.A = A
        self.filt.W = W

    def conv_to_steady_state(self):
        '''
        Create an SSKFDecoder object based on KalmanFilter parameters in this KFDecoder object
        '''
        from . import sskfdecoder
        self.filt = sskfdecoder.SteadyStateKalmanFilter(A=self.filt.A, W=self.filt.W, C=self.filt.C, Q=self.filt.Q) 

    def subselect_units(self, units):
        '''
        Prune units from the KFDecoder, e.g., due to loss of recordings for a particular cell

        Parameters
        units : string or np.ndarray of shape (N,2)
            The units which should be KEPT in the decoder

        Returns 
        -------
        KFDecoder 
            New KFDecoder object using only a subset of the cells of the original KFDecoder
        '''
        # Parse units into list of indices to keep
        inds_to_keep = self._proc_units(units, 'keep')
        dec_new = self._return_proc_units_decoder(inds_to_keep)
        return dec_new
        #self._save_new_dec(dec_new, '_subset')

def project_Q(C_v, Q_hat):
    """ 
    Deprecated! See clda.KFRML_IVC
    """
    print("projecting!")
    from scipy.optimize import fmin_bfgs, fmin_ncg

    C_v = np.mat(C_v)
    Q_hat = np.mat(Q_hat)
    Q_hat_inv = Q_hat.I

    c_1 = C_v[:,0]
    c_2 = C_v[:,1]
    A_1 = c_1*c_1.T - c_2*c_2.T
    A_2 = c_2*c_1.T
    A_3 = c_1*c_2.T
    A = [A_1, A_2, A_3]
    if 1:
        U = np.hstack([c_1 - c_2, c_2, c_1])
        V = np.vstack([(c_1 + c_2).T, c_1.T, c_2.T])
        C_inv_fn = lambda nu: np.mat(np.diag([1./nu[0], 1./(nu[0] + nu[1]), 1./(nu[2] - nu[0]) ]))
        C_fn = lambda nu: np.mat(np.diag([nu[0], (nu[0] + nu[1]), (nu[2] - nu[0]) ]))
        nu_0 = np.zeros(3)
        c_scalars = np.ones(3)
    else:
        u_1, s_1, v_1 = np.linalg.svd(A_1)
        c_scalars = np.hstack([s_1[0:2], 1, 1])
        U = np.hstack([u_1[:,0:2], c_2, c_1])
        V = np.vstack([v_1[0:2, :], c_1.T, c_2.T])
        C_fn = lambda nu: np.mat(np.diag(nu * c_scalars))
        nu_0 = np.zeros(4)

    def cost_fn_gen(nu, return_type='cost'):
        C = C_fn(nu)
        S_star_inv = Q_hat + U*C_fn(nu)*V
        #if return_type == 'cost':
        #    print C_v.T * S_star_inv * C_v
    
        if np.any(np.diag(C) == 0):
            S_star = S_star_inv.I
        else:
            C_inv = C.I
            S_star = Q_hat_inv - Q_hat_inv * U * (C_inv + V*Q_hat_inv*U).I*V * Q_hat_inv;
        
        # log-determinant using LU decomposition, required if Q is large, i.e. lots of simultaneous observations
        cost = -np.log(np.linalg.det(S_star_inv))
        #cost = -np.prod(np.linalg.slogdet(S_star_inv))
        
        # TODO gradient dimension needs to be the same as nu
        #grad = -np.array([np.trace(S_star*U[:,0] * c_scalars[0] * V[0,:]) for k in range(len(nu))])
        #grad = -1e-4*np.array([np.trace(S_star*A[0]), np.trace(S_star*A[1]), np.trace(S_star*A[2])])
        #print c_2.T*S_star*c_2
        grad = -1e-4*np.array(np.hstack([c_1.T*S_star*c_1 - c_2.T*S_star*c_2, c_1.T*S_star*c_2, c_2.T*S_star*c_1])).ravel()
        S = S_star
        hess = np.mat([[np.trace(S*A_1*S*A_1), np.trace(S*A_2*S*A_1), np.trace(S*A_3*S*A_1)],
                       [np.trace(S*A_1*S*A_2), np.trace(S*A_2*S*A_2), np.trace(S*A_3*S*A_2)],
                       [np.trace(S*A_1*S*A_3), np.trace(S*A_2*S*A_3), np.trace(S*A_3*S*A_3)]])
    
        #grad = hess*np.mat(grad.reshape(-1,1))
        #log = logging.getLogger()
        #print "nu = %s, cost = %g, grad=%s" % (nu, cost, grad)
        #log.warning("nu = %s, cost = %g, grad=%s" % (nu, cost, grad))
    
        if return_type == 'cost':
            return cost
        elif return_type == 'grad':
            return grad
        elif return_type == 'hess':
            return hess
        elif return_type == 'opt_val':
            return S_star
        else:
            raise ValueError("Cost function doesn't know how to return this: %s" % return_type)

    cost_fn = lambda nu: cost_fn_gen(nu, return_type = 'cost')
    grad    = lambda nu: cost_fn_gen(nu, return_type = 'grad')
    hess    = lambda nu: cost_fn_gen(nu, return_type = 'hess')
    arg_opt = lambda nu: cost_fn_gen(nu, return_type = 'opt_val')

    # Call optimization routine
    #v_star = fmin_ncg(cost_fn, nu_0, fprime=grad, fhess=hess, maxiter=10000)
    #print v_star
    #v_star = fmin_bfgs(cost_fn, nu_0, maxiter=10000, gtol=1e-15)
    v_star = fmin_bfgs(cost_fn, nu_0, fprime=grad, maxiter=10000, gtol=1e-15)
    print(v_star)

    Q_inv = arg_opt(v_star)
    Q = Q_inv.I
    Q = Q_hat + U * C_fn(v_star) * V

    # TODO print out (log) a more useful measure of success
    #print C_v.T * Q_inv * C_v
    #print C_v.T * Q.I * C_v
    #print v_star
    return Q




