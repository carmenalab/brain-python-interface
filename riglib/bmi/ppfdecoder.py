'''Needs docs'''

import numpy as np
from scipy.io import loadmat
from plexon import  psth

import tables
from itertools import izip

from riglib.bmi import bmi
from bmi import BMI, GaussianState
import statsmodels.api as sm # GLM fitting module

class PointProcessFilter():
    """Low-level KF, agnostic to application

    Model: 
       x_{t+1} = Ax_t + w_t; w_t ~ N(0, W)
       log(y_t) = Cx_t
    """

    def __init__(self, A, W, C, dt):
        self.A = np.mat(A)
        self.W = np.mat(W)
        self.C = np.mat(C)
        self.dt = dt
        
        self.state_noise = GaussianState(0.0, W)
        self._pickle_init()

    def _pickle_init(self):
        """Code common to unpickling and initialization
        """
        nS = self.A.shape[0]
        offset_row = np.zeros(nS)
        offset_row[-1] = 1
        self.include_offset = np.array_equal(np.array(self.A)[-1, :], offset_row)

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
        self.id = np.zeros([1, self.C.shape[1]])

    def __call__(self, obs):
        """ Call the 1-step forward inference function
        """
        self.state = self._forward_infer(self.state, obs)
        return self.state.mean

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def _check_valid(self, lambda_predict):
        if np.any((lambda_predict * self.dt) > 1): 
            raise ValueError("Cell exploded!")

    def _obs_prob(self, state):
        Loglambda_predict = self.C.T * state.mean
        lambda_predict = np.exp(Loglambda_predict)/self.dt

        #self._check_valid(self, lambda_predict, id)
        return lambda_predict
    
    def _ssm_pred(self, state):
        return self.A*state + self.state_noise

    def _forward_infer(self, st, obs_t): #stimulant_index, stoch_stim_index, stoch_index, det_index)
        obs_t = np.mat(obs_t.reshape(-1,1))
        C = self.C
        n_obs = C.shape[1]
        
        # TODO incorporate feedback control state space model
        pred_state = self._ssm_pred(st)
        assert pred_state.mean.shape == (3,1)
        pred_obs = self._obs_prob(pred_state)
        #print pred_obs

        D = np.mat(np.diag(np.array(pred_obs).ravel() * self.dt))
    
        P_pred = pred_state.cov
        nS = self.A.shape[0]
        if self.include_offset:
            P_pred_inv = np.mat(np.zeros([nS, nS]))
            P_pred_inv[:-1, :-1] = P_pred[:-1,:-1].I
            P_est = np.mat(np.zeros([nS, nS]))
            P_est[:-1, :-1] = (P_pred_inv[:-1, :-1] + C[:-1,:]*D*C[:-1,:].T).I
        else:
            P_est = (P_pred.I + C*D*C.T).I

        #import pdb
        #pdb.set_trace()

        unpred_spikes = obs_t - pred_obs*self.dt
        x_est = pred_state.mean + P_est*C*unpred_spikes
        post_state = GaussianState(x_est, P_est)
        assert post_state.mean.shape == (3,1)
        
        return post_state

    def __setstate__(self, state):
        """Set the model parameters {A, W, C, Q} stored in the pickled
        object"""
        self.A = state['A']
        self.W = state['W']
        self.C = state['C']
        self.dt = state['dt']
        self._pickle_init()

    def __getstate__(self):
        """Return the model parameters {A, W, C} for pickling"""
        return {'A':self.A, 'W':self.W, 'C':self.C, 'dt':self.dt}

    @classmethod
    def MLE_obs_model(cls, hidden_state, obs, include_offset=True):
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
        
        X = np.array(X)
        Y = np.array(Y)

        # ML estimate of C and Q
        n_units = Y.shape[0]
        C = np.zeros([3, n_units])
        glm_family = sm.families.Poisson()
        for k in range(n_units):
            model = sm.GLM(Y[k,:], X.T, family=glm_family)
            model_fit = model.fit()
            C[:,k] = model_fit.params

        return C,

class PPFDecoder(bmi.BMI, bmi.Decoder):
    def __init__(self, ppf, units, bounding_box, states, 
        states_to_bound, binlen=0.1, tslice=[-1,-1]):
        """ Initializes the Kalman filter decoder.  Includes BMI specific
        features used to run the Kalman filter in a BMI context.
        """
        self.ppf = ppf
        self.ppf._init_state()
        self.units = np.array(units, dtype=np.int32)
        self.binlen = binlen
        self.bin_spikes = psth.SpikeBin(self.units, self.binlen)
        self.bounding_box = bounding_box
        self.states = states
        self.tslice = tslice # TODO replace with real tslice
        self.states_to_bound = states_to_bound

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
        raise NotImplementedError

    def predict(self, ts_data_k, target=None, speed=0.05, target_radius=0.5,
                assist_level=0.9, dt=0.1, task_data=None):
        """Decode the spikes"""
        raise NotImplementedError

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
