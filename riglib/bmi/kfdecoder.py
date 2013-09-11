'''Needs docs'''

import pickle
import sys

import numpy as np
from scipy.io import loadmat
from plexon import plexfile, psth
from riglib.nidaq import parse

import tables
from itertools import izip
import logging

from . import BMI
python_plexnet_dtype = np.dtype([("ts", np.float), ("chan", np.int32), ("unit", np.int32)])

class GaussianState(object):
    def __init__(self, mean, cov):
        if isinstance(mean, float):
            self.mean = mean
        else:
            self.mean = np.mat(mean.reshape(-1,1))
        self.cov = np.mat(cov)
    
    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, np.float64) or isinstance(other, float):
            mu = other*self.mean
            cov = other**2 * self.cov
        elif isinstance(other, np.matrix):
            mu = other*self.mean
            cov = other*self.cov*other.T
        elif isinstance(other, np.ndarray):
            other = mat(array)
            mu = other*self.mean
            cov = other*self.cov*other.T
        else:
            print type(other)
            raise
        return GaussianState(mu, cov)

    def __mul__(self, other):
        mean = other*self.mean
        if isinstance(other, int) or isinstance(other, np.float64) or isinstance(other, float):
            cov = other**2 * self.cov
        else:
            print type(other)
            raise
        return GaussianState(mean, cov)

    def __add__(self, other):
        if isinstance(other, GaussianState):
            return GaussianState( self.mean+other.mean, self.cov+other.cov )


class KalmanFilter():
    """Low-level KF, agnostic to application

    Model: 
       x_{t+1} = Ax_t + w_t; w_t ~ N(0, W)
       y_t = Cx_t + q_t; q_t ~ N(0, Q)
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
        
        self.state_noise = GaussianState(0.0, W)
        self.obs_noise = GaussianState(0.0, Q)
        self._pickle_init()

    def _pickle_init(self):
        """Code common to unpickling and initialization
        """
        nS = self.A.shape[0]
        offset_row = np.zeros(nS)
        offset_row[-1] = 1
        self.include_offset = np.array_equal(np.array(self.A)[-1, :], offset_row)

        self.alt = nS < self.C.shape[0] # No. of states less than no. of observations
        if self.alt:
            C, Q = self.C, self.Q 
            self.C_xpose_Q_inv = C.T * Q.I
            self.C_xpose_Q_inv_C = C.T * Q.I * C
        else:
            self.C_xpose_Q_inv = None
            self.C_xpose_Q_inv_C = None

        try:
            self.is_stochastic
        except:
            n_states = self.A.shape[0]
            self.is_stochastic = np.ones(n_states, dtype=bool)

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
        self.obs_noise = GaussianState(0.0, self.Q)


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
        pred_obs = self._obs_prob(pred_state)

        C, Q = self.C, self.Q
        P = pred_state.cov

        K = self._calc_kalman_gain(P, **kwargs)
        I = np.mat(np.eye(self.C.shape[1]))

        post_state = pred_state
        post_state.mean += K*(obs_t - pred_obs.mean)
        post_state.cov = (I - K*C) * P 
        return post_state

    def forward_infer(self, st, obs_t):
        pred_state = self._ssm_pred(st)
        pred_obs = self._obs_prob(pred_state)

        H, Q = self.C, self.Q
        P_prior = pred_state.cov

        #if self.alt:
        #    K = P*( self.C_xpose_Q_inv - self.C_xpose_Q_inv_C*(P.I + self.C_xpose_Q_inv_C).I * self.C_xpose_Q_inv ) 
        #else:
        K = P_prior*H.T*np.linalg.pinv( H*P_prior*H.T + Q )
        I = np.mat(np.eye(self.C.shape[1]))

        pred_state.mean += K*(obs_t - pred_obs.mean)
        pred_state.cov = (I-K*H)*P_prior 
        return pred_state

    def _calc_kalman_gain(self, P, alt=False, verbose=False):
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

    def get_sskf(self, tol=1e-10, return_P=False, dtype=np.array, 
        verbose=False, return_Khist=False, alt=True):
        """ starting from the data in the decoder struct, compute the converged 
        Experimentally, convergence requires ~300 predictions. 10000 "predictions" performed
        by default, controlled by the kwarg 'n_steps'
        """ 
        A, W, C, Q = np.mat(self.A), np.mat(self.W), np.mat(self.C), np.mat(self.Q)

        P = np.mat( np.zeros(A.shape) )

        last_K = np.mat(np.ones(C.T.shape))*np.inf
        K = np.mat(np.ones(C.T.shape))*0

        K_hist = []

        iter_idx = 0
        while np.linalg.norm(K-last_K) > tol and iter_idx < 4000:
            P = A*P*A.T + W 
            last_K = K
            K = self._calc_kalman_gain(P, alt=alt, verbose=verbose)
            K_hist.append(K)
            P -= K*C*P;
            iter_idx += 1
        if verbose: print "Converged in %d iterations--error: %g" % (iter_idx, np.linalg.norm(K-last_K)) 
    
        n_state_vars, n_state_vars = A.shape
        A_bar = (np.mat(np.eye(n_state_vars, n_state_vars)) - K * C) * A
        B_bar = K 
    
        if return_P and return_Khist:
            return dtype(A_bar), dtype(B_bar), dtype(P), K_hist
        elif return_P:
            return dtype(A_bar), dtype(B_bar), dtype(P)
        elif return_Khist:
            return dtype(A_bar), dtype(B_bar), K_hist
        else:
            return dtype(A_bar), dtype(B_bar)

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
        self._pickle_init()

    def __getstate__(self):
        """Return the model parameters {A, W, C, Q} for pickling"""
        return {'A':self.A, 'W':self.W, 'C':self.C, 'Q':self.Q}
            #'alt':self.alt, 'C_xpose_Q_inv':self.C_xpose_Q_inv, 'C_xpose_Q_inv_C':self.C_xpose_Q_inv_C}

    @classmethod
    def MLE_obs_model(self, hidden_state, obs, include_offset=True, drives_obs=None):
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
        Q = np.cov( Y-C*X, bias=1 )
        if not drives_obs == None:
            n_obs = C.shape[0]
            C_tmp = np.zeros([n_obs, n_states])
            C_tmp[:,drives_obs] = C
            C = C_tmp
        return (C, Q)
    
    def get_params(self):
        return self.A, self.W, self.C, self.Q



class KFDecoder(BMI):
    def __init__(self, kf, mFR, sdFR, units, bounding_box, states, 
        states_to_bound, binlen=0.1, tslice=[-1,-1]):
        """ Initializes the Kalman filter decoder.  Includes BMI specific
        features used to run the Kalman filter in a BMI context.
        """
        self.kf = kf
        self.kf._init_state()
        self.mFR = mFR
        self.sdFR = sdFR
        self.zscore = False
        self.units = np.array(units, dtype=np.int32)
        self.binlen = binlen
        self.bin_spikes = psth.SpikeBin(self.units, self.binlen)
        self.bounding_box = bounding_box
        self.states = states
        self.tslice = tslice # TODO replace with real tslice
        self.states_to_bound = states_to_bound
        self.zeromeanunits = None

        # Gain terms for hack debugging
        try:
            f = open('/home/helene/bmi_gain', 'r')
            self.gain = [float(x) for x in f.readline().rstrip().split(',')]
            self.offset = [float(x) for x in f.readline().rstrip().split(',')]
        except:
            self.gain = 1
            self.offset = 0

    def init_zscore(self, mFR_curr, sdFR_curr):
        # if interfacing with Kinarm system, may mean and sd will be shape nx1
        self.zeromeanunits=np.nonzero(mFR_curr==0)[0] #find any units with a mean FR of zero for this session
        sdFR_curr[self.zeromeanunits]=np.nan # set mean and SD of quiet units to nan to avoid divide by 0 error
        mFR_curr[self.zeromeanunits]=np.nan
        self.sdFR_ratio = self.sdFR/sdFR_curr
        self.mFR_diff = mFR_curr-self.mFR
        self.zscore = True
        
    def bound_state(self):
        """Apply bounds on state vector, if bounding box is specified
        """
        if not self.bounding_box == None:
            min_bounds, max_bounds = self.bounding_box
            state = self[self.states_to_bound]
            repl_with_min = state < min_bounds
            state[repl_with_min] = min_bounds[repl_with_min]

            repl_with_max = state > max_bounds
            state[repl_with_max] = max_bounds[repl_with_max]
            self[self.states_to_bound] = state
            #self[self.states_] = 
            #horiz_min, vert_min, horiz_max, vert_max = self.bounding_box
            #self.kf.state.mean[0,0] = min(self.kf.state.mean[0,0], horiz_max)
            #self.kf.state.mean[0,0] = max(self.kf.state.mean[0,0], horiz_min)
            #self.kf.state.mean[1,0] = min(self.kf.state.mean[1,0], vert_max)
            #self.kf.state.mean[1,0] = max(self.kf.state.mean[1,0], vert_min)
    
    def __call__(self, obs_t, **kwargs):
        '''
        Return the predicted arm position given the new data.

        Parameters
        -----------
        newdata : array_like
            Recent spike data for all units

        Returns
        -------
        output : array_like
            Decoder output for each decoded parameter

        '''

        return self.predict(obs_t, **kwargs)


    def predict(self, ts_data_k, target=None, speed=1.0, target_radius=0.5,
                assist_level=0.0, dt=0.1, task_data=None, **kwargs):
        """Decode the spikes"""
        # Save the previous cursor state for assist
        prev_kin = self.kf.get_mean()
        if assist_level > 0 and not target == None:
            cursor_pos = prev_kin[0:2] # TODO assumes a 2D position state
            diff_vec = target - cursor_pos 
            dist_to_target = np.linalg.norm(diff_vec)
            dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
            
            if dist_to_target > target_radius:
                assist_cursor_pos = cursor_pos + speed*dir_to_target;
            else:
                assist_cursor_pos = cursor_pos + diff_vec/2;

            assist_cursor_vel = (assist_cursor_pos-cursor_pos)/dt;
            assist_cursor_kin = np.hstack([assist_cursor_pos, assist_cursor_vel, 1])

        # "Bin" spike timestamps to generate spike counts
        if len(ts_data_k) == 0:
            spike_counts = np.zeros((self.bin_spikes.nunits,))
            task_data['bin_edges'] = np.array([np.nan, np.nan])
        elif ts_data_k.dtype == python_plexnet_dtype:
            spike_counts = self.bin_spikes(ts_data_k)
        #    temp = np.array([e[0] for e in ts_data_k])
        #    task_data['bin_edges'] = np.array([np.min(temp), np.max(temp)])
        else:
            spike_counts = ts_data_k

        if task_data is not None:
            task_data['bins'] = spike_counts

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


        #add a scaling factor on the velocity to slow down or speed up the cursor. comment out the following lines to undo!
        # vel_scale_factor=.8
        # cursor_kin=self.kf.get_mean()
        # scaled_vel=cursor_kin[2:4]*vel_scale_factor
        # scaled_pos=prev_kin[0:2]+scaled_vel*dt
        # self.kf.state.mean[:,0] = np.hstack([scaled_pos, scaled_vel, 1]).reshape(-1,1)


        # Bound cursor, if applicable
        self.bound_state()

        if assist_level > 0 and not target == None:
            cursor_kin = self.kf.get_mean()
            kin = assist_level*assist_cursor_kin + (1-assist_level)*cursor_kin
            self.kf.state.mean[:,0] = kin.reshape(-1,1)
            self.bound_state()

        state = self.kf.get_mean()
        return np.array([state[0], 0, state[1], state[2], 0, state[3], 1])

    def retrain(self, batch, halflife):
        raise NotImplementedError

    def __getitem__(self, idx):
        """Get element(s) of the BMI state, indexed by name or number"""
        if isinstance(idx, int):
            return self.kf.state.mean[idx, 0]
        elif isinstance(idx, str) or isinstance(idx, unicode):
            idx = self.states.index(idx)
            return self.kf.state.mean[idx, 0]
        elif np.iterable(idx):
            return np.array([self.__getitem__(k) for k in idx])
        else:
            raise ValueError("KFDecoder: Improper index type: %" % type(idx))

    def __setitem__(self, idx, value):
        """Set element(s) of the BMI state, indexed by name or number"""
        if isinstance(idx, int):
            self.kf.state.mean[idx, 0] = value
        elif isinstance(idx, str) or isinstance(idx, unicode):
            idx = self.states.index(idx)
            self.kf.state.mean[idx, 0] = value
        elif np.iterable(idx):
            [self.__setitem__(k, val) for k, val in izip(idx, value)]
        else:
            raise ValueError("KFDecoder: Improper index type: %" % type(idx))

    def __setstate__(self, state):
        """Set decoder state after un-pickling"""
        self.bin_spikes = psth.SpikeBin(state['units'], state['binlen'])
        del state['cells']
        self.__dict__.update(state)
        self.kf._pickle_init()
        self.kf._init_state()

    def __getstate__(self):
        """Create dictionary describing state of the decoder instance, 
        for pickling"""
        state = dict(cells=self.units)
        exclude = set(['bin_spikes'])
        for k, v in self.__dict__.items():
            if k not in exclude:
                state[k] = v
        return state

    def get_state(self):
        return np.array(self.kf.state.mean).ravel()

    def update_params(self, new_params):
        C, Q, mFR, sdFR = new_params
        self.kf.C = C
        self.kf.Q = Q
        self.mFR=mFR
        self.sdFR=sdFR

def load_from_mat_file(decoder_fname, bounding_box=None, 
    states=['p_x', 'p_y', 'v_x', 'v_y', 'off'], states_to_bound=[]):
    """Create KFDecoder from MATLAB decoder file used in a Dexterit-based
    BMI
    """
    decoder_data = loadmat(decoder_fname)['decoder']
    A = decoder_data['A'][0,0]
    W = decoder_data['W'][0,0]
    H = decoder_data['H'][0,0]
    Q = decoder_data['Q'][0,0]
    mFR = decoder_data['mFR'][0,0]
    sdFR = decoder_data['sdFR'][0,0]

    pred_sigs = [str(x[0]) for x in decoder_data['predSig'][0,0].ravel()]
    unit_lut = {'a':1, 'b':2, 'c':3, 'd':4}
    units = [(int(sig[3:6]), unit_lut[sig[-1]]) for sig in pred_sigs]

    kf = KalmanFilter(A, W, H, Q)
    kfdecoder = KFDecoder(kf, mFR, sdFR, units, bounding_box, states, states_to_bound)

    return kfdecoder

def project_Q(C_v, Q_hat):
    """ Constrain Q such that the first two columns of the H matrix
    are independent and have identical gain in the steady-state KF

    TODO next: implement without using the math trick
    """
    print "projecting!"
    from scipy.optimize import fmin_bfgs

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
    
        if np.any(np.diag(C) == 0):
            S_star = S_star_inv.I
        else:
            C_inv = C.I
            S_star = Q_hat_inv - Q_hat_inv * U * (C_inv + V*Q_hat_inv*U).I*V * Q_hat_inv;
        
        # log-determinant using LU decomposition, required if Q is large, i.e. lots of simultaneous observations
        cost = -np.log(np.linalg.det(S_star_inv))
        #cost = -np.prod(np.linalg.slogdet(S_star_inv))
        
        # TODO gradient dimension needs to be the same as nu
        grad = -np.array([np.trace(S_star*U[:,0] * c_scalars[0] * V[0,:]) for k in range(len(nu))])
        grad = -1e-8*np.array([np.trace(S_star*A[0]), np.trace(S_star*A[1]), np.trace(S_star*A[2])])
    
        #log = logging.getLogger()
        #print "nu = %s, cost = %g, grad=%s" % (nu, cost, grad)
        #log.warning("nu = %s, cost = %g, grad=%s" % (nu, cost, grad))
    
        if return_type == 'cost':
            return cost
        elif return_type == 'grad':
            return grad
        elif return_type == 'opt_val':
            return S_star
        else:
            raise ValueError("Cost function doesn't know how to return this: %s" % return_type)

    cost_fn = lambda nu: cost_fn_gen(nu, return_type = 'cost')
    grad    = lambda nu: cost_fn_gen(nu, return_type = 'grad')
    arg_opt = lambda nu: cost_fn_gen(nu, return_type = 'opt_val')

    # Call optimization routine
    v_star = fmin_bfgs(cost_fn, nu_0, fprime=grad, maxiter=10000, gtol=1e-15)

    Q_inv = arg_opt(v_star)
    Q = Q_inv.I
    Q = Q_hat + U * C_fn(v_star) * V

    # TODO print out (log) a more useful measure of success
    print C_v.T * Q_inv * C_v
    print C_v.T * Q.I * C_v
    print v_star
    return Q

if __name__ == '__main__':
    cells = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1)]
    block = 'cart20130428_01'
    files = dict(plexon='/storage/plexon/%s.plx' % block, hdf='/storage/rawdata/hdf/%s.hdf' % block)
    train_from_manual_control(cells, **files)
    pass
