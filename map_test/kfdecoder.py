import pickle
import sys

import numpy as np
from scipy.io import loadmat
from plexon import psth

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

        nS = self.A.shape[0]
        offset_row = np.zeros(nS)
        offset_row[-1] = 1
        self.include_offset = np.array_equal(np.array(A)[-1, :], offset_row)

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

    def _set_alt(self):
        # TODO alt method or not should be determiend automatically
        raise NotImplementedError
        try:
            C = self.C
            Q = self.Q 
            self.C_xpose_Q_inv = C.T * Q.I
            self.C_xpose_Q_inv_C = C.T * Q.I * C
        except:
            pass
        
    def _forward_infer(self, st, obs_t):
        pred_state = self._ssm_pred(st)
        pred_obs = self._obs_prob(pred_state)

        C, Q = self.C, self.Q
        P_prior = pred_state.cov

        if False: # TODO
            K = P*( self.C_xpose_Q_inv - self.C_xpose_Q_inv_C*(P.I + self.C_xpose_Q_inv_C).I * self.C_xpose_Q_inv ) 
        else:
            K = P_prior*C.T*np.linalg.pinv( C*P_prior*C.T + Q )
        I = np.mat(np.eye(self.C.shape[1]))

        post_state = pred_state
        post_state.mean += K*(obs_t - pred_obs.mean)
        post_state.cov = (I - K*C) * P_prior 
        return post_state

    def get_sskf(self, tol=1e-10, return_P=False, alt=False, dtype=np.array, 
        verbose=False, return_Khist=False):
        """ starting from the data in the decoder struct, compute the converged 
        Experimentally, convergence requires ~300 predictions. 10000 "predictions" performed
        by default, controlled by the kwarg 'n_steps'
        """ 
        A, W, H, Q = np.mat(self.A), np.mat(self.W), np.mat(self.H), np.mat(self.Q)

        if alt:
            H_xpose_Q_inv = H.T * Q.I
            H_xpose_Q_inv_H = H.T * Q.I * H
        
        P = np.mat( np.zeros(A.shape) )

        last_K = np.mat(np.ones(H.T.shape))*np.inf
        K = np.mat(np.ones(H.T.shape))*0

        K_hist = []

        iter_idx = 0
        while np.linalg.norm(K-last_K) > tol and iter_idx < 4000:
            P = A*P*A.T + W 
            last_K = K
            if alt or self.alt:
                K = P*( H_xpose_Q_inv - H_xpose_Q_inv_H*(P.I + H_xpose_Q_inv_H).I * H_xpose_Q_inv ) 
            else:
                K = (P*H.T)*np.linalg.pinv(H*P*H.T + Q);
            K_hist.append(K)
            P -= K*H*P;
            iter_idx += 1
        if verbose: print "Converged in %d iterations--error: %g" % (iter_idx, np.linalg.norm(K-last_K)) 
    
        n_state_vars, n_state_vars = A.shape
        A_bar = (np.mat(np.eye(n_state_vars, n_state_vars)) - K * H) * A
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

    def __getstate__(self):
        """Return the model parameters {A, W, C, Q} for pickling"""
        return {'A':self.A, 'W':self.W, 'C':self.C, 'Q':self.Q}

class KFDecoder():
    def __init__(self, kf, mFR, sdFR, units, bounding_box):
        self.kf = kf
        self.kf._init_state()
        self.mFR = mFR
        self.sdFR = sdFR
        self.zscore = False
        self.units = np.array(units, dtype=np.int32)
        self.bin_spikes = psth.SpikeBin(self.units, np.inf)
        self.bounding_box = bounding_box

    def init_zscore(self, mFR_curr, sdFR_curr):
        self.sdFR_ratio = np.ravel(self.sdFR/sdFR_curr)
        self.mFR = mFR_curr.ravel() # overwrite the original mean firing rate
        self.zscore = not np.all(self.sdFR_ratio == 1)
        
    def load(self, decoder_fname):
        kf = pickle.load(open(decoder_fname, 'rb'))
        return kf
    load = classmethod(load)

    def decode(self, ts_data_k, target=None, assist_level=0):
        # "Bin" spike timestamps to generate spike counts
        spike_counts = self.bin_spikes(ts_data_k)

        # re-normalize the variance of the spike observations, if nec
        if self.zscore:
            spike_counts = (spike_counts - self.mFR) * self.sdFR_ratio

        # re-format as a 1D col vec
        spike_counts = np.mat(spike_counts.reshape(-1,1))

        # Run the KF
        self.kf(spike_counts)

        # Bound cursor, if applicable
        if not self.bounding_box == None:
            horiz_min, vert_min, horiz_max, vert_max = self.bounding_box
            self.kf.state.mean[0,0] = min(self.kf.state.mean[0,0], horiz_max)
            self.kf.state.mean[0,0] = max(self.kf.state.mean[0,0], horiz_min)
            self.kf.state.mean[1,0] = min(self.kf.state.mean[1,0], vert_max)
            self.kf.state.mean[1,0] = max(self.kf.state.mean[1,0], vert_min)

        return self.kf.get_mean()

    def retrain(self, batch, halflife):
        raise NotImplementedError


def load_from_mat_file(decoder_fname, bounding_box=None):
    # Create KF decoder from MATLAB stored file (KINARM bw compatibilty)
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
    kfdecoder = KFDecoder(kf, mFR, sdFR, units, bounding_box)
    
    return kfdecoder

if __name__ == '__main__':
    pass
