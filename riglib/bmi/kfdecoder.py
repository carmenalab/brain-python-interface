import pickle
import sys

import numpy as np
from scipy.io import loadmat
from plexon import plexfile, psth
from riglib.nidaq import parse

import tables
from itertools import izip

from . import BMI

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

    def __init__(self, A, W, C, Q):
        self.A = np.mat(A)
        self.W = np.mat(W)
        self.C = np.mat(C)
        self.Q = np.mat(Q)
        
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


    def __call__(self, obs):
        """ Call the 1-step forward inference function
        """
        self.state = self._forward_infer(self.state, obs)
        return self.state.mean

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def _obs_prob(self, state):
        return self.C * state + self.obs_noise

    def _ssm_pred(self, state):
        return self.A*state + self.state_noise

    def _forward_infer(self, st, obs_t):
        pred_state = self._ssm_pred(st)
        pred_obs = self._obs_prob(pred_state)

        C, Q = self.C, self.Q
        P = pred_state.cov

        K = self._calc_kalman_gain(P)
        I = np.mat(np.eye(self.C.shape[1]))

        post_state = pred_state
        post_state.mean += K*(obs_t - pred_obs.mean)
        post_state.cov = (I - K*C) * P 
        return post_state

    def _calc_kalman_gain(self, P, alt=True, verbose=False):
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

    def MLE_obs_model(self, hidden_state, obs, include_offset=True):
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
    
        # ML estimate of C and Q
        C = Y*np.linalg.pinv(X)
        Q = np.cov( Y-C*X, bias=1 )
        return (C, Q)
    MLE_obs_model = classmethod(MLE_obs_model)

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

        # Gain terms for hack debugging
        try:
            f = open('/home/helene/bmi_gain', 'r')
            self.gain = [float(x) for x in f.readline().rstrip().split(',')]
            self.offset = [float(x) for x in f.readline().rstrip().split(',')]
        except:
            self.gain = 1
            self.offset = 0

    def init_zscore(self, mFR_curr, sdFR_curr):
        self.sdFR_ratio = np.ravel(self.sdFR/sdFR_curr)
        self.mFR = mFR_curr.ravel() # overwrite the original mean firing rate
        self.zscore = not np.all(self.sdFR_ratio == 1)
        
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

    def predict(self, ts_data_k, target=None, speed=0.05, target_radius=0.5,
                assist_level=0.9, dt=0.1, task_data=None):
        """Decode the spikes"""
        # Save the previous cursor state if using assist
        if assist_level > 0 and not target == None:
            prev_kin = self.kf.get_mean()
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
        spike_counts = self.bin_spikes(ts_data_k)

        # re-normalize the variance of the spike observations, if nec
        if self.zscore:
            spike_counts = (spike_counts - self.mFR) * self.sdFR_ratio

        if task_data is not None:
            task_data['bins'] = spike_counts

        # re-format as a 1D col vec
        spike_counts = np.mat(spike_counts.reshape(-1,1))

        # Run the KF
        self.kf(spike_counts)

        # Bound cursor, if applicable
        self.bound_state()

        if assist_level > 0 and not target == None:
            cursor_kin = self.kf.get_mean()
            kin = assist_level*assist_cursor_kin + (1-assist_level)*cursor_kin
            self.kf.state.mean[:,0] = kin.reshape(-1,1)
            self.bound_state()

        # TODO manual gain and offset terms
        # f = open('/home/helene/bmi_gain', 'r')
        # gain = [float(x) for x in f.readline().rstrip().split(',')]
        # offset = [float(x) for x in f.readline().rstrip().split(',')]
        # pt[1] = 0
        # pt[0] = (pt[0] + offset[0])*gain[0]
        # pt[2] = (pt[2] + offset[2])*gain[2]

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

if __name__ == '__main__':
    cells = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1)]
    block = 'cart20130428_01'
    files = dict(plexon='/storage/plexon/%s.plx' % block, hdf='/storage/rawdata/hdf/%s.hdf' % block)
    train_from_manual_control(cells, **files)
    pass
