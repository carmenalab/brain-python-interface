'''
Closed-loop decoder adaptation (CLDA) classes. There are two types of classes,
"Learners" and "Updaters". Learners implement various methods to estimate the
"intended" BMI movements of the user. Updaters implement various method for 
updating 
'''
import multiprocessing as mp
import numpy as np
from . import kfdecoder, ppfdecoder, train, bmi, feedback_controllers
import time
import cmath
from itertools import izip
import tables
import re
import assist
import os

from utils.angle_utils import *

inv = np.linalg.inv

try:
    from numpy.linalg import lapack_lite
    lapack_routine = lapack_lite.dgesv
except:
    pass
    
def fast_inv(A):
    '''
    This method represents a way to speed up matrix inverse computations when 
    several independent matrix inverses of all the same shape must be taken all
    at once. This is used by the PPFContinuousBayesianUpdater. Without this method,
    the updates could not be performed in real time with ~30 cells (compute complexity 
    is linear in the number of units, so it is possible that fewer units would not
    have had this issue).

    Code stolen from: 
    # http://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy 
    '''
    b = np.identity(A.shape[2], dtype=A.dtype)

    n_eq = A.shape[1]
    n_rhs = A.shape[2]
    pivots = np.zeros(n_eq, np.intc)
    identity  = np.eye(n_eq)
    def lapack_inverse(a):
        b = np.copy(identity)
        pivots = np.zeros(n_eq, np.intc)
        results = lapack_lite.dgesv(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
        if results['info'] > 0:
            raise np.LinAlgError('Singular matrix')
        return b

    return np.array([lapack_inverse(a) for a in A])

def slow_inv(A):
    return np.array([np.linalg.inv(a) for a in A])

##############################################################################
## Learners
##############################################################################
class Learner(object):
    '''
    Classes for estimating the 'intention' of the BMI operator, inferring the intention from task goals.
    '''
    def __init__(self, batch_size, *args, **kwargs):
        '''
        Instantiate a Learner for estimating intention during CLDA

        Parameters
        ----------
        batch_size: int
            number of samples used to estimate each new decoder parameter setting
        done_states: list of strings, optional
            states of the task which end a batch, regardless of the length of the batch. default = []
        reset_states: list of strings, optional
            states of the task which, if encountered, reset the batch regardless of its length. default = []

        '''
        self.done_states = kwargs.pop('done_states', [])
        self.reset_states = kwargs.pop('reset_states', [])
        print "Reset states for learner: "
        print self.reset_states
        print "Done states for learner: "
        print self.done_states        
        self.batch_size = batch_size
        self.passed_done_state = False
        self.enabled = True
        self.input_state_index = -1
        self.reset()

    def disable(self):
        '''Set a flag to disable forming intention estimates from new incoming data'''
        self.enabled = False

    def enable(self):
        '''Set a flag to enable forming intention estimates from new incoming data'''
        self.enabled = True

    def reset(self):
        '''Reset the lists of saved intention estimates and corresponding neural data'''
        self.kindata = []
        self.neuraldata = []
        self.obs_value = []

    def __call__(self, spike_counts, decoder_state, target_state, decoder_output, task_state, state_order=None, **kwargs):
        """
        Calculate the intended kinematics and pair with the neural data

        Parameters
        ----------
        spike_counts : np.mat of shape (K, 1)
            Neural observations used to decode 'decoder_state'
        decoder_state : np.mat of shape (N, 1)
            State estimate output from the decoder
        target_state : np.mat of shape (N, 1)
            For the current time, this is the optimal state for the Decoder as specified by the task
        decoder_output : np.mat of shape (N, 1)
            ... this seems like the same as decoder_state
        task_state : string
            Name of the task state; some learners (e.g., the cursorGoal learner) have different intention estimates depending on the phase of the task/trial
        state_order : np.ndarray of shape (N,), optional
            Order of each state in the decoder; see riglib.bmi.state_space_models.State
        **kwargs: dict
            Optional keyword arguments for the 'value' calculator

        Returns
        -------
        None
        """
        if task_state in self.reset_states:
            print "resetting CLDA batch"
            self.reset()

        int_kin = self.calc_int_kin(decoder_state, target_state, decoder_output, task_state, state_order=state_order)
        obs_value = self.calc_value(decoder_state, target_state, decoder_output, task_state, state_order=state_order, **kwargs)
        
        if self.passed_done_state and self.enabled:
            if task_state in ['hold', 'target']:
                self.passed_done_state = False

        if self.enabled and not self.passed_done_state and int_kin is not None:
            self.kindata.append(int_kin)
            self.neuraldata.append(spike_counts)
            self.obs_value.append(obs_value)

            if task_state in self.done_states:
                self.passed_done_state = True

    def calc_value(self, *args, **kwargs):
        '''
        Calculate a "value", i.e. a usefulness, for a particular observation. 
        Can override in child classe for RL-style updates, but a priori all observations are equally informative
        '''
        return 1.

    def is_ready(self):
        '''
        Returns True if the collected estimates of the subject's intention are ready for processing into new decoder parameters
        '''
        _is_ready = len(self.kindata) >= self.batch_size or ((len(self.kindata) > 0) and self.passed_done_state)
        return _is_ready

    def get_batch(self):
        '''
        Returns all the data from the last 'batch' of obserations of intended kinematics and neural decoder inputs
        '''
        kindata = np.hstack(self.kindata)
        neuraldata = np.hstack(self.neuraldata)
        self.reset()
        return dict(intended_kin=kindata, spike_counts=neuraldata)
        # return kindata, neuraldata

class DumbLearner(Learner):
    '''
    A learner that never learns anything. Used to make non-adaptive BMI tasks interface the same as CLDA tasks.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for DumbLearner

        Parameters
        ----------
        args, kwargs: positional and keyword arguments
            Ignored, none are needed

        Returns
        -------
        DumbLearner instance
        '''
        self.enabled = False
        self.input_state_index = 0

    def __call__(self, *args, **kwargs):
        """
        Do nothing; hence the name of the class

        Parameters
        ----------
        args, kwargs: positional and keyword arguments
            Ignored, none are needed

        Returns
        -------
        None
        """
        pass

    def is_ready(self):
        '''DumbLearner is never ready to tell you what it learnerd'''
        return False

    def get_batch(self):
        '''DumbLearner never has any 'batch' data to retrieve'''
        raise NotImplementedError

class FeedbackControllerLearner(Learner):
    '''
    An intention estimator where the subject is assumed to operate like a state feedback controller
    '''
    def __init__(self, batch_size, fb_ctrl, *args, **kwargs):
        self.fb_ctrl = fb_ctrl
        self.style = kwargs.pop('style', 'mixing')
        super(FeedbackControllerLearner, self).__init__(batch_size, *args, **kwargs)

    def calc_int_kin(self, current_state, target_state, decoder_output, task_state, state_order=None):
        try:
            if self.style == 'additive':
                output = self.fb_ctrl(current_state, target_state, mode=task_state)
            elif self.style == 'mixing':
                output = self.fb_ctrl.calc_next_state(current_state, target_state, mode=task_state)
            return output
        except:
            # Key errors happen when the feedback controller doesn't have a policy for the current task state
            import traceback
            traceback.print_exc()
            return None


class OFCLearner(Learner):
    '''
    An intention estimator where the subject is assumed to operate like a muiti-modal LQR controller
    '''
    def __init__(self, batch_size, A, B, F_dict, *args, **kwargs):
        '''
        Constructor for OFCLearner

        Parameters
        ----------
        batch_size : int
            size of batch of samples to pass to the Updater to estimate new decoder parameters
        A : np.mat 
            State transition matrix of the modeled discrete-time system
        B : np.mat 
            Control input matrix of the modeled discrete-time system
        F_dict : dict
            Keys match names of task states, values are feedback matrices (size n_inputs x n_states)
        *args : additional comma-separated args
            Passed to super constructor
        **kwargs : additional keyword args
            Passed to super constructor

        Returns
        -------
        OFCLearner instance
        '''
        super(OFCLearner, self).__init__(batch_size, *args, **kwargs)
        self.B = B
        self.F_dict = F_dict
        self.A = A

    def calc_int_kin(self, current_state, target_state, decoder_output, task_state, state_order=None):
        '''
        Calculate intended kinematics as 
            x_t^{int} = A*x_t + B*F(x^* - x_t)

        Parameters
        ----------
        current_state : np.mat of shape (N, 1)
            State estimate output from the decoder.
        target_state : np.mat of shape (N, 1)
            For the current time, this is the optimal state for the Decoder as specified by the task
        decoder_output : np.mat of shape (N, 1)
            State estimate output from the decoder, after the current observations (may be one step removed from 'current_state')
        task_state : string
            Name of the task state; some learners (e.g., the cursorGoal learner) have different intention estimates depending on the phase of the task/trial
        state_order : np.ndarray of shape (N,), optional
            Order of each state in the decoder; see riglib.bmi.state_space_models.State

        Returns
        -------
        np.mat of shape (N, 1)
            Estimate of intended next state for BMI
        '''
        if 0:
            print "OFCLearner" 
        try:
            current_state = np.mat(current_state).reshape(-1,1)
            target_state = np.mat(target_state).reshape(-1,1)
            F = self.F_dict[task_state]
            A = self.A
            B = self.B

            if 0:
                print "current_state"
                print current_state
                print "target_state"
                print target_state

            return A*current_state + B*F*(target_state - current_state)
        except KeyError:
            return None

class RegexKeyDict(dict):
    '''
    Dictionary where key matching applies regular expressions in addition to exact matches
    '''
    def __getitem__(self, key):
        '''
        Lookup key in dictionary by finding exactly one dict key which, by regex, matches the input argument 'key'
        '''
        keys = self.keys()
        matching_keys = filter(lambda x: re.match(x, key), keys)
        if len(matching_keys) == 0:
            raise KeyError("No matching keys were found!")
        elif len(matching_keys) > 1:
            raise ValueError("Multiple keys match!")
        else:
            return super(RegexKeyDict, self).__getitem__(matching_keys[0])

    def __contains__(self, key):
        '''
        Determine if a key is in the dictionary using regular expression matching
        '''
        keys = self.keys()
        matching_keys = filter(lambda x: re.match(x, key), keys)
        if len(matching_keys) == 0:
            return False
        elif len(matching_keys) > 1:
            raise ValueError("Multiple keys match!")
        else:
            return True        

##############################################################################
## Updaters
##############################################################################
from riglib.mp_calc import MPCompute
class Updater(object):
    '''
    Wrapper for MPCompute computations running in another process
    '''
    def __init__(self, fn, multiproc=False, verbose=False):
        self.verbose = verbose
        self.multiproc = multiproc
        if self.multiproc:
            # create the queues
            self.work_queue = mp.Queue()
            self.result_queue = mp.Queue()

            # Instantiate the process
            self.calculator = MPCompute(self.work_queue, self.result_queue, fn)

            # spawn the process
            self.calculator.start()
        else:
            self.fn = fn

        self._result = None
        self.waiting = False

    def init(self, decoder):
        pass

    def __call__(self, *args, **kwargs):
        input_data = (args, kwargs)
        if self.multiproc:
            if self.verbose: print "queuing job"
            self.work_queue.put(input_data)    
            self.prev_input = input_data
            self.waiting = True
        else:
            self._result = self.fn(*args, **kwargs)

    def get_result(self):
        if self.multiproc:
            try:
                output_data = self.result_queue.get_nowait()
                self.prev_result = output_data
                self.waiting = False
                return output_data
            except Queue.Empty:
                return None
            except:
                import traceback
                traceback.print_exc()
        else:
            res = self._result
            self._result = None
            return res

    def __del__(self):
        '''
        Stop the child process if one was spawned
        '''
        if self.multiproc:
            self.calculator.stop()


class KFSmoothbatch(Updater):
    '''
    Calculate KF Parameter updates using the SmoothBatch method. See [Orsborn et al, 2012] for mathematical details
    '''
    update_kwargs = dict(steady_state=True)
    def __init__(self, batch_time, half_life):
        '''    Docstring    '''
        super(KFSmoothbatch, self).__init__(self.calc, multiproc=False)
        self.half_life = half_life
        self.batch_time = batch_time
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))
        
    def calc(self, intended_kin=None, spike_counts=None, decoder=None, half_life=None, **kwargs):
        """
        Smoothbatch calculations

        Run least-squares on (intended_kinematics, spike_counts) to 
        determine the C_hat and Q_hat of new batch. Then combine with 
        old parameters using step-size rho
        """
        print "calculating new SB parameters"
        C_old          = decoder.kf.C
        Q_old          = decoder.kf.Q
        drives_neurons = decoder.drives_neurons
        mFR_old        = decoder.mFR
        sdFR_old       = decoder.sdFR

        C_hat, Q_hat = kfdecoder.KalmanFilter.MLE_obs_model(
            intended_kin, spike_counts, include_offset=False, drives_obs=drives_neurons)

        if not (half_life is None):
            rho = np.exp(np.log(0.5)/(half_life/self.batch_time))
        else:
            rho = self.rho 

        C = (1-rho)*C_hat + rho*C_old
        Q = (1-rho)*Q_hat + rho*Q_old

        mFR = (1-rho)*np.mean(spike_counts.T, axis=0) + rho*mFR_old
        sdFR = (1-rho)*np.std(spike_counts.T, axis=0) + rho*sdFR_old
        
        D = C.T * np.linalg.pinv(Q) * C
        new_params = {'kf.C':C, 'kf.Q':Q, 
            'kf.C_xpose_Q_inv_C':D, 'kf.C_xpose_Q_inv':C.T * np.linalg.pinv(Q),
            'mFR':mFR, 'sdFR':sdFR}
        return new_params


class KFOrthogonalPlantSmoothbatch(KFSmoothbatch):
    '''    Docstring    '''
    def __init__(self, *args, **kwargs):
        '''    
        Docstring    
        '''
        self.default_gain = kwargs.pop('default_gain', None)
        suoer(KFOrthogonalPlantSmoothbatch, self).__init__(*args, **kwargs)

    def calc(self, *args, **kwargs):
        '''    Docstring    '''
        new_params = super(KFOrthogonalPlantSmoothbatch, self).calc(*args, **kwargs)
        C, Q, = new_params['kf.C'], new_params['kf.Q']

        D = (C.T * np.linalg.pinv(Q) * C)
        if self.default_gain == None:
            d = np.mean([D[3,3], D[5,5]])
            D[3:6, 3:6] = np.diag([d, d, d])
        else:
            # calculate the gain from the riccati equation solution
            A_diag = np.diag(np.asarray(decoder.filt.A[3:6, 3:6]))
            W_diag = np.diag(np.asarray(decoder.filt.W[3:6, 3:6]))
            D_diag = []
            for a, w, n in izip(A_diag, W_diag, self.default_gain):
                d = self.scalar_riccati_eq_soln(a, w, n)
                D_diag.append(d)

            D[3:6, 3:6] = np.mat(np.diag(D_diag))

        new_params['kf.C_xpose_Q_inv_C'] = D
        new_params['kf.C_xpose_Q_inv'] = C.T * np.linalg.pinv(Q)
        return new_params


class PPFSmoothbatch(Updater):
    '''    Docstring    '''
    def __init__(self, batch_time, half_life):
        '''    Docstring    '''
        super(PPFSmoothbatch, self).__init__(self.calc, multiproc=True)
        self.half_life = half_life
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))

    def calc(self, intended_kin=None, spike_counts=None, decoder=None, half_life=None, **kwargs):
        """
        Smoothbatch calculations

        Run least-squares on (intended_kinematics, spike_counts) to 
        determine the C_hat and Q_hat of new batch. Then combine with 
        old parameters using step-size rho
        """

        if half_life is not None:
            rho = np.exp(np.log(0.5)/(half_life/self.batch_time))
        else:
            rho = self.rho 

        C_old = decoder.filt.C
        drives_neurons = decoder.drives_neurons
        states = decoder.states
        decoding_states = np.take(states, np.nonzero(drives_neurons)).ravel().tolist() #['hand_vx', 'hand_vz', 'offset'] 

        C_hat, pvalues = ppfdecoder.PointProcessFilter.MLE_obs_model(
            intended_kin, spike_counts, include_offset=False, drives_obs=drives_neurons)
        C_hat = train.inflate(C_hat, decoding_states, states, axis=1)
        pvalues = train.inflate(pvalues, decoding_states, states, axis=1)
        pvalues[pvalues[:,:-1] == 0] = np.inf

        mesh = np.nonzero(pvalues < 0.1)
        C = np.array(C_old.copy())
        C[mesh] = (1-rho)*C_hat[mesh] + rho*np.array(C_old)[mesh]
        C = np.mat(C)

        new_params = {'filt.C':C}
        return new_params

class PPFContinuousBayesianUpdater(Updater):
    '''
    Adapt the parameters of a PPFDecoder using an HMM to implement a gradient-descent type parameter update.

    (currently only works for PPFs which do not also include the self-history or correlational elements)
    '''
    update_kwargs = dict()
    def __init__(self, decoder, units='cm', param_noise_scale=1., param_noise_variances=None):
        '''
        Constructor for PPFContinuousBayesianUpdater

        Parameters
        ----------
        decoder : bmi.ppfdecoder.PPFDecoder instance
            Should have a 'filt' attribute which is a PointProcessFilter instance
        units : string
            Docstring
        param_noise_scale : float
            Multiplicative factor to increase the parameter "process noise". Higher values result in faster but less stable parameter convergence.
        '''
        super(PPFContinuousBayesianUpdater, self).__init__(self.calc, multiproc=False)

        self.n_units = decoder.filt.C.shape[0]
        if param_noise_variances == None:
            if units == 'm':
                vel_gain = 1e-4
            elif units == 'cm':
                vel_gain = 1e-8

            print "Updater param noise scale %g" % param_noise_scale
            vel_gain *= param_noise_scale
            param_noise_variances = np.array([vel_gain*0.13, vel_gain*0.13, 1e-4*0.06/50])
        self.W = np.tile(np.diag(param_noise_variances), [self.n_units, 1, 1])


        self.P_params_est = self.W.copy()

        self.neuron_driving_state_inds = np.nonzero(decoder.drives_neurons)[0]
        self.neuron_driving_states = list(np.take(decoder.states, np.nonzero(decoder.drives_neurons)[0]))
        self.n_states = len(decoder.states)
        self.full_size = len(decoder.states)

        self.dt = decoder.filt.dt
        self.beta_est = np.array(decoder.filt.C)

    def calc(self, intended_kin=None, spike_counts=None, decoder=None, **kwargs):
        '''    Docstring    '''

        if (intended_kin is None) or (spike_counts is None) or (decoder is None):
            raise ValueError("must specify intended_kin, spike_counts and decoder objects for the updater to work!")        

        if 0:
            print np.array(intended_kin).ravel()

        int_kin_full = intended_kin
        spike_obs_full = spike_counts
        n_samples = int_kin_full.shape[1]

        # Squash any observed spike counts which are greater than 1
        spike_obs_full[spike_obs_full > 1] = 1
        for k in range(n_samples):
            spike_obs = spike_obs_full[:,k]
            int_kin = int_kin_full[:,k]

            beta_est = self.beta_est[:,self.neuron_driving_state_inds]
            int_kin = np.asarray(int_kin).ravel()[self.neuron_driving_state_inds]
            Loglambda_predict = np.dot(int_kin, beta_est.T)
            rates = np.exp(Loglambda_predict)
            if np.any(rates > 1):
                print 'rates > 1!'
                rates[rates > 1] = 1
            unpred_spikes = np.asarray(spike_obs).ravel() - rates

            C_xpose_C = np.outer(int_kin, int_kin)

            self.P_params_est += self.W
            try:
                P_params_est_inv = fast_inv(self.P_params_est)
            except:
                P_params_est_inv = slow_inv(self.P_params_est)
            L = np.dstack([rates[c] * C_xpose_C for c in range(self.n_units)]).transpose([2,0,1])

            try:
                self.P_params_est = fast_inv(P_params_est_inv + L)
            except:
                self.P_params_est = slow_inv(P_params_est_inv + L)

            beta_est += (unpred_spikes * np.dot(int_kin, self.P_params_est).T).T

            # store beta_est
            self.beta_est[:, self.neuron_driving_state_inds] = beta_est

        return {'filt.C': np.mat(self.beta_est.copy())}


class KFRML(Updater):
    '''
    Calculate updates for KF parameters using the recursive maximum likelihood (RML) method
    See (Dangi et al, Neural Computation, 2014) for mathematical details.
    '''
    update_kwargs = dict(steady_state=False)
    def __init__(self, batch_time, half_life, adapt_C_xpose_Q_inv_C=True):
        '''
        Constructor for KFRML

        Parameters
        ----------
        batch_time : float
            Size of data batch to use for each update. Specify in seconds.
        half_life : float 
            Amount of time (in seconds) before parameters are half-overwritten by new data.
        adapt_C_xpose_Q_inv_C : bool
            Flag specifying whether to update the decoder property C^T Q^{-1} C, which 
            defines the feedback dynamics of the final closed-loop system if A and W are known

        Returns
        -------
        KFRML instance
        '''
        super(KFRML, self).__init__(self.calc, multiproc=False)
        # self.work_queue = None
        self.batch_time = batch_time
        # self.result_queue = None        
        self.half_life = half_life
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))
        self.adapt_C_xpose_Q_inv_C = adapt_C_xpose_Q_inv_C

        self._new_params = None

    @staticmethod
    def compute_suff_stats(hidden_state, obs, include_offset=True):
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
        assert hidden_state.shape[1] == obs.shape[1]
    
        if isinstance(hidden_state, np.ma.core.MaskedArray):
            mask = ~hidden_state.mask[0,:] # NOTE THE INVERTER 
            inds = np.nonzero([ mask[k]*mask[k+1] for k in range(len(mask)-1)])[0]
    
            X = np.mat(hidden_state[:,mask])
            n_pts = len(np.nonzero(mask)[0])
    
            Y = np.mat(obs[:,mask])
            if include_offset:
                X = np.vstack([ X, np.ones([1,n_pts]) ])
        else:
            num_hidden_state, n_pts = hidden_state.shape
            X = np.mat(hidden_state)
            if include_offset:
                X = np.vstack([ X, np.ones([1,n_pts]) ])
            Y = np.mat(obs)
        X = np.mat(X, dtype=np.float64)

        R = (X * X.T)
        S = (Y * X.T)
        T = (Y * Y.T)
        ESS = n_pts

        return (R, S, T, ESS)

    def init(self, decoder):
        '''
        Retrieve sufficient statistics from the seed decoder.

        Parameters
        ----------
        decoder : bmi.Decoder instance
            The seed decoder before any adaptation runs.

        Returns
        -------
        None
        '''
        self.R = decoder.filt.R
        self.S = decoder.filt.S
        self.T = decoder.filt.T
        self.ESS = decoder.filt.ESS

        self.feature_inds = np.arange(decoder.n_features)

        # By default, tuning parameters for all features will adapt
        self.stable_inds = []
        self.adapting_inds = self.feature_inds.copy()
        self.adapting_inds_mesh = np.ix_(self.adapting_inds, self.adapting_inds)
        self.stable_inds_independent = False

    def calc(self, intended_kin=None, spike_counts=None, decoder=None, half_life=None, values=None, **kwargs):
        '''
        Parameters
        ----------
        intended_kin : np.ndarray of shape (n_states, batch_size)
            Batch of estimates of intended kinematics, from the learner
        spike_counts : np.ndarray of shape (n_features, batch_size)
            Batch of observations of decoder features, from the learner
        decoder : bmi.Decoder instance
            Reference to the Decoder instance
        half_life : float, optional
            Half-life to use to calculate the parameter change step size. If not specified, the half-life specified when the Updater was constructed is used.
        values : np.ndarray, optional
            Relative value of each sample of the batch. If not specified, each sample is assumed to have equal value.
        kwargs : dict
            Optional keyword arguments, ignored

        Returns
        -------
        new_params : dict
            New parameters to feed back to the Decoder in use by the task.
        '''
        if intended_kin == None or spike_counts == None or decoder == None:
            raise ValueError("must specify intended_kin, spike_counts and decoder objects for the updater to work!")

        # Calculate the step size based on the half life and the number of samples to train from
        batch_size = intended_kin.shape[1]
        batch_time = batch_size * decoder.binlen            

        if half_life is not None:
            rho = np.exp(np.log(0.5)/(half_life/batch_time))
        else:
            rho = self.rho 

        drives_neurons = decoder.drives_neurons
        mFR_old        = decoder.mFR
        sdFR_old       = decoder.sdFR

        x = np.mat(intended_kin)
        y = np.mat(spike_counts)

        # limit y to the features which are permitted to adapt
        y = y[self.adapting_inds, :]

        if values is not None:
            n_samples = np.sum(values)
            B = np.mat(np.diag(values))
        else:
            n_samples = spike_counts.shape[1]
            B = np.mat(np.eye(n_samples))

        if self.adapt_C_xpose_Q_inv_C:
            self.R = rho*self.R + (x*B*x.T)

        self.S[self.adapting_inds,:] = rho*self.S[self.adapting_inds,:] + (y*B*x.T)
        self.T[self.adapting_inds_mesh] = rho*self.T[self.adapting_inds_mesh] + np.dot(y, B*y.T)
        self.ESS = rho*self.ESS + n_samples

        R_inv = np.mat(np.zeros(self.R.shape))
        R_inv[np.ix_(drives_neurons, drives_neurons)] = self.R[np.ix_(drives_neurons, drives_neurons)].I
        C = self.S * R_inv

        Q = (1./self.ESS) * (self.T - self.S*C.T)
        if hasattr(self, 'stable_inds_mesh'):
            Q[self.stable_inds_mesh] = decoder.filt.Q[self.stable_inds_mesh]
        if self.stable_inds_independent:
            Q[np.ix_(self.stable_inds, self.adapting_inds)] = 0
            Q[np.ix_(self.adapting_inds, self.stable_inds)] = 0

        mFR = (1-rho)*np.mean(spike_counts.T, axis=0) + rho*mFR_old
        sdFR = (1-rho)*np.std(spike_counts.T, axis=0) + rho*sdFR_old

        C_xpose_Q_inv = C.T * np.linalg.pinv(Q)

        new_params = {'filt.C':C, 'filt.Q':Q, 'filt.C_xpose_Q_inv':C_xpose_Q_inv,
            'mFR':mFR, 'sdFR':sdFR, 'kf.ESS':self.ESS, 'filt.S':self.S, 'filt.T':self.T}

        if self.adapt_C_xpose_Q_inv_C:
            C_xpose_Q_inv_C = C_xpose_Q_inv * C
            new_params['filt.C_xpose_Q_inv_C'] = C_xpose_Q_inv_C
            new_params['filt.R'] = self.R
        else:
            new_params['filt.C_xpose_Q_inv_C'] = decoder.filt.C_xpose_Q_inv_C
            new_params['filt.R'] = decoder.filt.R

        self._new_params = new_params
        return new_params

    def set_stable_inds(self, stable_inds, stable_inds_independent=False):
        '''
        Docstring
        '''
        self.stable_inds = stable_inds
        self.adapting_inds = np.array(filter(lambda x: x not in self.stable_inds, self.feature_inds))
        self.adapting_inds_mesh = np.ix_(self.adapting_inds, self.adapting_inds)
        self.stable_inds_mesh = np.ix_(self.stable_inds, self.stable_inds)
        self.stable_inds_independent = stable_inds_independent


class PPFRML(Updater):
    '''
    '''
    update_kwargs = dict()
    def __init__(self, *args, **kwargs):
        super(PPFRML, self).__init__(self.calc, multiproc=False)

    def init(self, decoder):
        self.dt = decoder.filt.dt
        self.C_est = decoder.filt.C
        self.H = decoder.H
        self.M = decoder.M
        self.S = decoder.S
        
        self.neuron_driving_state_inds = np.nonzero(decoder.drives_neurons)[0]
        self.neuron_driving_states = list(np.take(decoder.states, np.nonzero(decoder.drives_neurons)[0]))
        self.n_states = len(decoder.states)
        self.full_size = len(decoder.states)
        

    def calc(self, intended_kin=None, spike_counts=None, decoder=None, half_life=120., **kwargs):
        '''
        # time iterative RLS
        '''
        if (intended_kin is None) or (spike_counts is None) or (decoder is None):
            raise ValueError("must specify intended_kin, spike_counts and decoder objects for the updater to work!")        

        batch_size = 1.
        batch_time = batch_size * decoder.binlen    
        rho = np.exp(np.log(0.5)/(half_life/batch_time))
    
        n_cells = self.C_est.shape[0]
        n_obs = intended_kin.shape[1]
        intended_kin = np.mat(intended_kin)
        # print "updating"
        # print intended_kin
        # print spike_counts.T
        spike_counts[spike_counts > 1] = 1
        for k in range(n_cells):
            for m in range(n_obs):
                H = np.mat(self.H[k])
                S = np.mat(self.S[k].reshape(-1,1))
                M = np.mat(self.M[k].reshape(-1,1))
                c = self.C_est[k, self.neuron_driving_state_inds].T

                # print H
                c_new = c - H.I * (S - M)
                c = rho*c + (1-rho)*c_new
                self.C_est[k, self.neuron_driving_state_inds] = c.T
                
                x_m = intended_kin[self.neuron_driving_state_inds, m] #X[k].T
                mu_m = np.exp(c.T * x_m)[0,0]
                y_m = spike_counts[k, m]


                self.H[k] = rho*H + (1-rho)*(-mu_m * x_m * x_m.T)
                self.M[k] = np.array(rho*M + (1-rho)*(mu_m * x_m)).ravel()
                self.S[k] = np.array(rho*S + (1-rho)*(y_m*x_m)).ravel()

        return {'filt.C': self.C_est}


class KFRML_IVC(KFRML):
    '''
    RML version where diagonality constraints are imposed on the steady state KF matrices
    '''
    default_gain = None
    def calc(self, intended_kin=None, spike_counts=None, decoder=None, half_life=None, values=None, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        new_params = super(KFRML_IVC, self).calc(intended_kin=intended_kin, spike_counts=spike_counts, decoder=decoder, half_life=half_life, values=values, **kwargs)
        C, Q, = new_params['filt.C'], new_params['filt.Q']

        D = (C.T * np.linalg.pinv(Q) * C)
        if self.default_gain == None:
            d = np.mean([D[3,3], D[5,5]])
            D[3:6, 3:6] = np.diag([d, d, d])
        else:
            # calculate the gain from the riccati equation solution
            A_diag = np.diag(np.asarray(decoder.filt.A[3:6, 3:6]))
            W_diag = np.diag(np.asarray(decoder.filt.W[3:6, 3:6]))
            D_diag = []
            for a, w, n in izip(A_diag, W_diag, [self.default_gain]*3):
                d = self.scalar_riccati_eq_soln(a, w, n)
                D_diag.append(d)

            D[3:6, 3:6] = np.mat(np.diag(D_diag))

        new_params['filt.C_xpose_Q_inv_C'] = D
        new_params['filt.C_xpose_Q_inv'] = C.T * np.linalg.pinv(Q)
        return new_params

    @classmethod
    def scalar_riccati_eq_soln(cls, a, w, n):
        '''    Docstring    '''
        return (1-a*n)/w * (a-n)/n         


class KFRML_baseline(KFRML):
    '''    Docstring    '''
    def calc(self, intended_kin, spike_counts, decoder, half_life=None, **kwargs):
        '''    Docstring    '''

        if half_life is not None:
            rho = np.exp(np.log(0.5)/(half_life/self.batch_time))
        else:
            rho = self.rho 

        drives_neurons = decoder.drives_neurons
        mFR_old        = decoder.mFR
        sdFR_old       = decoder.sdFR

        x = intended_kin
        y = spike_counts

        self.R = rho*self.R + (x*x.T)
        self.S = rho*self.S + (y*x.T)
        self.T = rho*self.T + np.dot(y, y.T) #(y*y.T)
        self.ESS = rho*self.ESS + 1

        R_inv = np.mat(np.zeros(self.R.shape))
        R_inv[np.ix_(drives_neurons, drives_neurons)] = self.R[np.ix_(drives_neurons, drives_neurons)].I
        C_new = self.S * R_inv

        # Q = 1./(1-(rho)**self.iter_counter) * (self.T - self.S*C.T)
        Q = decoder.filt.Q

        mFR = (1-rho)*np.mean(spike_counts.T,axis=0) + rho*mFR_old
        sdFR = (1-rho)*np.std(spike_counts.T,axis=0) + rho*sdFR_old

        C = decoder.filt.C
        C[:,-1] = C_new[:,-1]

        C_xpose_Q_inv   = C.T * np.linalg.pinv(Q)
        C_xpose_Q_inv_C = C_xpose_Q_inv * C
        
        new_params = {'kf.C':C, 'kf.Q':Q, 
            'kf.C_xpose_Q_inv_C':C_xpose_Q_inv_C, 'kf.C_xpose_Q_inv':C_xpose_Q_inv,
            'mFR':mFR, 'sdFR':sdFR}

        return new_params



def write_clda_data_to_hdf_table(hdf_fname, data, ignore_none=False):
    '''
    Save CLDA data generated during the experiment to the specified HDF file

    Parameters
    ----------
    hdf_fname : string
        filename of HDF file
    data : list
        list of dictionaries with the same keys and same dtypes for values

    Returns
    -------
    None
    '''
    log_file = open(os.path.expandvars('$HOME/code/bmi3d/log/clda_hdf_log'), 'w')

    compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)
    if len(data) > 0:
        # Find the first parameter update dictionary
        k = 0
        first_update = data[k]
        while first_update is None:
            k += 1
            first_update = data[k]
    
        table_col_names = first_update.keys()
        print table_col_names
        dtype = []
        shapes = []
        for col_name in table_col_names:
            if isinstance(first_update[col_name], float):
                shape = (1,)
            else:
                shape = first_update[col_name].shape
            dtype.append((col_name.replace('.', '_'), 'f8', shape))
            shapes.append(shape)
    
        log_file.write(str(dtype))
        # Create the HDF table with the datatype above
        dtype = np.dtype(dtype) 
    
        h5file = tables.openFile(hdf_fname, mode='a')
        arr = h5file.createTable("/", 'clda', dtype, filters=compfilt)

        null_update = np.zeros((1,), dtype=dtype)
        for col_name in table_col_names:
            null_update[col_name.replace('.', '_')] *= np.nan
    
        for k, param_update in enumerate(data):
            log_file.write('%d, %s\n' % (k, str(ignore_none)))
            if param_update == None:
                if ignore_none:
                    continue
                else:
                    data_row = null_update
            else:
                data_row = np.zeros((1,), dtype=dtype)
                for col_name in table_col_names:
                    data_row[col_name.replace('.', '_')] = np.asarray(param_update[col_name])
    
            arr.append(data_row)
        h5file.close()
        

if __name__ == '__main__':
    # Test case for CLDARecomputeParameters, to show non-blocking properties
    # of the recomputation
    work_queue = mp.Queue()
    result_queue = mp.Queue()

    work_queue.put((None, None, None))

    clda_worker = CLDARecomputeParameters(work_queue, result_queue)
    clda_worker.start()

    while 1:
        try:
            result = result_queue.get_nowait()
            break
        except:
            print 'stuff'
        time.sleep(0.1)
