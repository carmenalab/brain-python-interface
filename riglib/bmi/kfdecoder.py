'''
Implementation of a Kalman filter and associated code to use it as a BMI
decoder
'''

try:
    from plexon import psth
except:
    import warnings
    warnings.warn('psth module not found, using python spike binning function')

import numpy as np
from scipy.io import loadmat

import bmi
class KalmanFilter(bmi.GaussianStateHMM):
    """Low-level KF, agnostic to application

    Model: 
       x_{t+1} = Ax_t + w_t;   w_t ~ N(0, W)
           y_t = Cx_t + q_t;   q_t ~ N(0, Q)
    """

    def __init__(self, A, W, C, Q, is_stochastic=None):
        self.A = np.mat(A)
        self.W = np.mat(W)
        self.C = np.mat(C)
        self.Q = np.mat(Q)

        if is_stochastic == None:
            n_states = A.shape[0]
            self.is_stochastic = np.ones(n_states, dtype=bool)
        else:
            self.is_stochastic = is_stochastic
        
        self.state_noise = bmi.GaussianState(0.0, W)
        self.obs_noise = bmi.GaussianState(0.0, Q)
        self._pickle_init()

    def _pickle_init(self):
        """Code common to unpickling and initialization
        """
        nS = self.A.shape[0]
        offset_row = np.zeros(nS)
        offset_row[-1] = 1
        self.include_offset = np.array_equal(np.array(self.A)[-1, :], offset_row)

        self.alt = nS < self.C.shape[0] # No. of states less than no. of observations
        attrs = self.__dict__.keys()
        if not 'C_xpose_Q_inv_C' in attrs:
            C, Q = self.C, self.Q 
            self.C_xpose_Q_inv = C.T * Q.I
            self.C_xpose_Q_inv_C = C.T * Q.I * C

        try:
            self.is_stochastic
        except:
            n_states = self.A.shape[0]
            self.is_stochastic = np.ones(n_states, dtype=bool)

    def __call__(self, obs, **kwargs):
        """ Call the 1-step forward inference function
        """
        self.state = self._forward_infer(self.state, obs, **kwargs)
        return self.state.mean

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def _obs_prob(self, state):
        return self.C * state + self.obs_noise

    def _ssm_pred(self, state):
        return self.A*state + self.state_noise

    def _forward_infer(self, st, obs_t, **kwargs):
        pred_state = self._ssm_pred(st)
        #pred_obs = self._obs_prob(pred_state)

        C, Q = self.C, self.Q
        P = pred_state.cov

        K = self._calc_kalman_gain(P, **kwargs)
        #K = self._calc_kalman_gain2(P, **kwargs)
        I = np.mat(np.eye(self.C.shape[1]))
        D = self.C_xpose_Q_inv_C
        KC = P*(I - D*P*(I + D*P).I)*D

        post_state = pred_state
        ##print KC.shape
        ##print pred_state.mean.shape
        ##print K.shape
        ##print obs_t.shape
        ##print -KC*pred_state.mean + K*obs_t
        post_state.mean += -KC*pred_state.mean + K*obs_t
        post_state.cov = (I - KC) * P 
        return post_state

    def _calc_kalman_gain(self, P, alt=False, verbose=False):
        '''
        Deprecated version of Kalman gain computation
        Function below is always feasible and almost always faster
        '''
        A, W, C, Q = np.mat(self.A), np.mat(self.W), np.mat(self.C), np.mat(self.Q)
        if self.alt and alt:
            try:
                # print "trying alt method"
                if self.include_offset:
                    tmp = np.mat(np.zeros(self.A.shape))
                    tmp[:-1,:-1] = (P[:-1,:-1].I + self.C_xpose_Q_inv_C[:-1,:-1]).I
                else:
                    tmp = (P.I + self.C_xpose_Q_inv_C).I
                K = P*( self.C_xpose_Q_inv - self.C_xpose_Q_inv_C*tmp* self.C_xpose_Q_inv ) 
            except:
                if verbose: print "reverting"
                K = P*C.T*np.linalg.pinv( C*P*C.T + Q )
        else:
            K = P*C.T*np.linalg.pinv( C*P*C.T + Q )
        K[~self.is_stochastic, :] = 0
        return K

    def _calc_kalman_gain(self, P, verbose=False):
        '''
        Calculate Kalman gain using the alternate definition
        '''
        nX = P.shape[0]
        I = np.mat(np.eye(nX))
        D = self.C_xpose_Q_inv_C
        L = self.C_xpose_Q_inv
        K = P * (I - D*P*(I + D*P).I) * L
        return K

    def get_sskf(self, tol=1e-10, return_P=False, dtype=np.array, 
        verbose=False, return_Khist=False, alt=True):
        """ starting from the data in the decoder struct, compute the converged 
        Experimentally, convergence requires ~300 predictions. 10000 "predictions" performed
        by default, controlled by the kwarg 'n_steps'
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
        while np.linalg.norm(K-last_K) > tol and iter_idx < 4000:
            P = A*P*A.T + W 
            last_K = K
            K = self._calc_kalman_gain(P) #, alt=alt, verbose=verbose)
            K_hist.append(K)
            KC = P*(I - D*P*(I + D*P).I)*D
            P -= KC*P;
            iter_idx += 1
        if verbose: print "Converged in %d iterations--error: %g" % (iter_idx, np.linalg.norm(K-last_K)) 
    
        n_state_vars, n_state_vars = A.shape
        F = (np.mat(np.eye(n_state_vars, n_state_vars)) - KC) * A
    
        if return_P and return_Khist:
            return dtype(F), dtype(K), dtype(P), K_hist
        elif return_P:
            return dtype(F), dtype(K), dtype(P)
        elif return_Khist:
            return dtype(F), dtype(K), K_hist
        else:
            return dtype(F), dtype(K)

    def get_kalman_gain_seq(self, N=1000, tol=1e-10, verbose=False):
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
                    if verbose: print "breaking after %d iterations" % n

        return K, ss_idx

    def get_kf_system_mats(self, T):
        """KF system matrices
        """
        F = [None]*T
        K, ss_idx = self.get_kalman_gain_seq(N=T, verbose=False)
        nX = self.kf.A.shape[0]
        I = np.mat(np.eye(nX))
        
        for t in range(T):
            if t > ss_idx: F[t] = F[ss_idx]
            else: F[t] = (I - K[t]*self.kf.H)*self.kf.A
        
        return F, K

    def __setstate__(self, state):
        """Set the model parameters {A, W, C, Q} stored in the pickled
        object"""
        self.A = state['A']
        self.W = state['W']
        self.C = state['C']
        self.Q = state['Q']

        try:
            self.C_xpose_Q_inv_C = state['C_xpose_Q_inv_C']
            self.C_xpose_Q_inv = state['C_xpose_Q_inv']

            self.R = state['R']
            self.S = state['S']
            self.T = state['T']            
        except:
            # handled by _pickle_init
            pass

        self._pickle_init()

    def __getstate__(self):
        """Return the model parameters {A, W, C, Q} for pickling"""
        data = dict(A=self.A, W=self.W, C=self.C, Q=self.Q, 
                    C_xpose_Q_inv=self.C_xpose_Q_inv, 
                    C_xpose_Q_inv_C=self.C_xpose_Q_inv_C)
        data['R'] = self.R
        data['S'] = self.S
        data['T'] = self.T
        return data

    @classmethod
    def MLE_obs_model(self, hidden_state, obs, include_offset=True, 
                      drives_obs=None):
        """Unconstrained ML estimator of {C, Q} given observations and
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
            Y = np.mat(obs)
    
        n_states = X.shape[0]
        if not drives_obs == None:
            X = X[drives_obs, :]
            
        # ML estimate of C and Q
        
        C = np.mat(np.linalg.lstsq(X.T, Y.T)[0].T)
        #C = Y*np.linalg.pinv(X)
        Q = np.cov(Y - C*X, bias=1)
        if not drives_obs == None:
            n_obs = C.shape[0]
            C_tmp = np.zeros([n_obs, n_states])
            C_tmp[:,drives_obs] = C
            C = C_tmp
        return (C, Q)
    
    def get_params(self):
        return self.A, self.W, self.C, self.Q

    def set_steady_state_pred_cov(self):
        A, W, C, Q = np.mat(self.A), np.mat(self.W), np.mat(self.C), np.mat(self.Q)
        D = self.C_xpose_Q_inv_C 

        nS = A.shape[0]
        P = np.mat(np.zeros([nS, nS]))
        I = np.mat(np.eye(nS))

        last_K = np.mat(np.ones(C.T.shape))*np.inf
        K = np.mat(np.ones(C.T.shape))*0

        iter_idx = 0
        for iter_idx in range(40):
        #while iter_idx < 400:
        #while np.linalg.norm(K-last_K) > tol and iter_idx < 4000:
            P = A*P*A.T + W
            last_K = K
            KC = P*(I - D*P*(I + D*P).I)*D
            P -= KC*P;

        # TODO fix
        P[0:3, 0:3] = 0
        F, K = self.get_sskf()
        F = (I - KC)*A
        self._init_state(init_state=self.state.mean, init_cov=P)


class KFDecoder(bmi.BMI, bmi.Decoder):
    def __init__(self, kf, mFR, sdFR, units, bounding_box, states, drives_neurons,
        states_to_bound, binlen=0.1, n_subbins=1, tslice=[-1,-1]):
        """ 
        Initializes the Kalman filter decoder.  Includes BMI specific
        features used to run the Kalman filter in a BMI context.
        """
        self.kf = kf
        self.kf._init_state()
        self.mFR = mFR
        self.sdFR = sdFR
        self.zscore = False
        self.units = np.array(units, dtype=np.int32)
        self.binlen = binlen
        #self.bin_spikes = psth.SpikeBin(self.units, self.binlen)
        self.bounding_box = bounding_box
        self.states = states
        self.tslice = tslice # Legacy from when it was assumed that all decoders would be trained from manual control
        self.states_to_bound = states_to_bound
        self.zeromeanunits = None
        self.drives_neurons = drives_neurons
        self.n_subbins = n_subbins


        self.bmicount = 0
        self.bminum = int(self.binlen/(1/60.0))
        self.spike_counts = np.zeros([len(units), 1])

    def init_zscore(self, mFR_curr, sdFR_curr):
        # if interfacing with Kinarm system, may mean and sd will be shape nx1
        self.zeromeanunits=np.nonzero(mFR_curr==0)[0] #find any units with a mean FR of zero for this session
        sdFR_curr[self.zeromeanunits]=np.nan # set mean and SD of quiet units to nan to avoid divide by 0 error
        mFR_curr[self.zeromeanunits]=np.nan
        self.sdFR_ratio = self.sdFR/sdFR_curr
        self.mFR_diff = mFR_curr-self.mFR
        self.zscore = True
        
    def __call__(self, obs_t, **kwargs):
        '''
        Return the predicted arm position given the new data.
        '''
        self.spike_counts += obs_t.reshape(-1, 1)
        if self.bmicount == self.bminum-1:  
            self.bmicount = 0
            self.predict(self.spike_counts, **kwargs)
            self.spike_counts = np.zeros([len(self.units), 1])
        else:
            self.bmicount += 1
        return self.kf.get_mean()

    def predict(self, spike_counts, target=None, speed=0.5, target_radius=2,
                assist_level=0.0, assist_inds=[0,1,2],
                **kwargs):
        """Decode the spikes"""
        # Save the previous cursor state for assist
        prev_kin = self.kf.get_mean()
        if assist_level > 0:
            cursor_pos = prev_kin[assist_inds]
            diff_vec = target - cursor_pos 
            dist_to_target = np.linalg.norm(diff_vec)
            dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
            
            if dist_to_target > target_radius:
                assist_cursor_pos = cursor_pos + speed*dir_to_target
            else:
                assist_cursor_pos = cursor_pos + speed*diff_vec/2

            assist_cursor_vel = (assist_cursor_pos-cursor_pos)/self.binlen
            assist_cursor_kin = np.hstack([assist_cursor_pos, assist_cursor_vel, 1])

        # re-normalize the variance of the spike observations, if nec
        if self.zscore:
            spike_counts = (spike_counts - self.mFR_diff) * self.sdFR_ratio
            # set the spike count of any unit with a 0 mean to it's original mean
            # This is essentially removing it from the decoder.
            spike_counts[self.zeromeanunits] = self.mFR[self.zeromeanunits] 

        # re-format as a 1D col vec
        spike_counts = np.mat(spike_counts.reshape(-1,1))

        # Run the KF
        self.kf(spike_counts)

        # Bound cursor, if any hard bounds for states are applied
        self.bound_state()

        if assist_level > 0:
            cursor_kin = self.kf.get_mean()
            kin = assist_level*assist_cursor_kin + (1-assist_level)*cursor_kin
            self.kf.state.mean[:,0] = kin.reshape(-1,1)
            self.bound_state()

        state = self.kf.get_mean()
        return state

    def get_filter(self):
        return self.kf
        
    def get_state(self):
        '''
        Get the state of the decoder (mean of the Gaussian RV representing the
        state of the BMI)
        '''
        alg = self.get_filter()
        return np.array(alg.state.mean).ravel()

    def update_params(self, new_params):
        super(KFDecoder, self).update_params(new_params)

        # set the KF to the new steady state
        self.kf.set_steady_state_pred_cov()

    def __setstate__(self, state):
        """
        Set decoder state after un-pickling
        """
        super(KFDecoder, self).__setstate__(state)
        self.bmicount = 0
        self.bminum = int(self.binlen/(1/60.0))


def project_Q(C_v, Q_hat):
    """ Constrain Q such that the first two columns of the H matrix
    are independent and have identical gain in the steady-state KF

    TODO next: implement without using the math trick
    """
    print "projecting!"
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
    print v_star

    Q_inv = arg_opt(v_star)
    Q = Q_inv.I
    Q = Q_hat + U * C_fn(v_star) * V

    # TODO print out (log) a more useful measure of success
    #print C_v.T * Q_inv * C_v
    #print C_v.T * Q.I * C_v
    #print v_star
    return Q
