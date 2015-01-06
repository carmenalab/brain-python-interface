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

from state_space_models import StateSpaceArmAssist, StateSpaceReHand, StateSpaceIsMore
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

##############################################################################
## Learners
##############################################################################
def normalize(vec):
    '''
    Vector normalization. If the vector to be normalized is of norm 0, a vector of 0's is returned

    Parameters
    ----------
    vec: np.ndarray of shape (N,) or (N, 1)
        Vector to be normalized

    Returns
    -------
    norm_vec: np.ndarray of shape matching 'vec'
        Normalized version of vec
    '''
    norm_vec = vec / np.linalg.norm(vec)
    
    if np.any(np.isnan(norm_vec)):
        norm_vec = np.zeros_like(vec)
    
    return norm_vec

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
            State estimate output from the decoder.
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
            ... this seems like the same as decoder_state
        task_state : string
            Name of the task state; some learners (e.g., the cursorGoal learner) have different intention estimates depending on the phase of the task/trial
        state_order : np.ndarray of shape (N,), optional
            Order of each state in the decoder; see riglib.bmi.state_space_models.State

        Returns
        -------
        np.mat of shape (N, 1)
            Estimate of intended next state for BMI
        '''
        try:
            current_state = np.mat(current_state).reshape(-1,1)
            target_state = np.mat(target_state).reshape(-1,1)
            F = self.F_dict[task_state]
            A = self.A
            B = self.B
            return A*current_state + B*F*(target_state - current_state)        
        except KeyError:
            return None

class OFCLearner3DEndptPPF(OFCLearner):
    '''
    Specific instance of the OFCLearner for a PPF-controlled cursor
    '''
    def __init__(self, batch_size, *args, **kwargs):
        '''
        TODO to generalize this better, should be able to store these objects
        to file, just like a decoder, since the feedback matrices may change 
        on different days...
        '''
        dt = kwargs.pop('dt', 1./180)
        use_tau_unNat = kwargs.pop('tau', 2.7)
        self.tau = use_tau_unNat
        print "learner cost fn param: %g" % use_tau_unNat
        tau_scale = 28*use_tau_unNat/1000
        bin_num_ms = (dt/0.001)
        w_r = 3*tau_scale**2/2*(bin_num_ms)**2*26.61
        
        I = np.eye(3)
        zero_col = np.zeros([3, 1])
        zero_row = np.zeros([1, 3])
        zero = np.zeros([1,1])
        one = np.ones([1,1])
        A = np.bmat([[I, dt*I, zero_col], 
                     [0*I, 0*I, zero_col], 
                     [zero_row, zero_row, one]])
        B = np.bmat([[0*I], 
                     [dt/1e-3 * I],
                     [zero_row]])
        Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = np.mat(np.diag([w_r, w_r, w_r]))
        
        F = feedback_controllers.LQRController.dlqr(A, B, Q, R)
        F_dict = dict(target=F, hold=F) 
        super(OFCLearner3DEndptPPF, self).__init__(batch_size, A, B, F_dict, *args, **kwargs)

        # Tell BMISystem that this learner wants the most recent output
        # of the decoder rather than the second most recent, to matcn MATLAB
        self.input_state_index = 0

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
        keys = self.keys()
        matching_keys = filter(lambda x: re.match(x, key), keys)
        if len(matching_keys) == 0:
            return False
        elif len(matching_keys) > 1:
            raise ValueError("Multiple keys match!")
        else:
            return True        

class OFCLearnerTentacle(OFCLearner):
    '''    Docstring    '''
    def __init__(self, batch_size, A, B, Q, R, *args, **kwargs):
        '''    Docstring    '''
        F = feedback_controllers.LQRController.dlqr(A, B, Q, R)
        F_dict = RegexKeyDict()
        # F_dict['target'] = F
        # F_dict['hold'] = F
        F_dict['.*'] = F
        super(OFCLearnerTentacle, self).__init__(batch_size, A, B, F_dict, *args, **kwargs)

class TentacleValueLearner(Learner):
    _mean = 24.5
    _mean_alpha = 0.99
    def __init__(self, *args, **kwargs):
        if 'kin_chain' not in kwargs:
            raise ValueError("kin_chain object must specified for TentacleValueLearner!")
        self.kin_chain = kwargs.pop('kin_chain')
        super(TentacleValueLearner, self).__init__(*args, **kwargs)

        dt = 0.1
        use_tau_unNat = 2.7
        tau = use_tau_unNat
        tau_scale = 28*use_tau_unNat/1000
        bin_num_ms = (dt/0.001)
        w_r = 3*tau_scale**2/2*(bin_num_ms)**2*26.61

        I = np.eye(3)
        zero_col = np.zeros([3, 1])
        zero_row = np.zeros([1, 3])
        zero = np.zeros([1,1])
        one = np.ones([1,1])
        A = self.A = np.bmat([[I, dt*I, zero_col], 
                     [0*I, 0*I, zero_col], 
                     [zero_row, zero_row, one]])
        B = self.B = np.bmat([[0*I], 
                     [dt/1e-3 * I],
                     [zero_row]])
        Q = self.Q = np.mat(np.diag([1., 1, 1, 0, 0, 0, 0]))
        R = self.R = np.mat(np.diag([w_r, w_r, w_r]))

        self.F = feedback_controllers.LQRController.dlqr(A, B, Q, R)

    def calc_int_kin(self, current_state, target_state, decoder_output, task_state, state_order=None):
        '''
        This method of intention estimation just uses the subject's output 
        '''
        return decoder_output.reshape(-1,1)

    def calc_value(self, current_state, target_state, decoder_output, task_state, state_order=None, horizon=10, **kwargs):
        '''
        Determine the 'value' of a tentacle movement (4-link arm) 
        '''
        current_state = np.array(current_state).ravel()
        target_state = np.array(target_state).ravel()
        
        joint_pos = current_state[:self.kin_chain.n_links]
        endpt_pos = self.kin_chain.endpoint_pos(joint_pos)
        J = self.kin_chain.jacobian(-joint_pos)
        joint_vel = current_state[4:8] ### TODO remove hardcoding
        endpt_vel = np.dot(J, joint_vel)
        current_state_endpt = np.hstack([endpt_pos, endpt_vel[0], 0, endpt_vel[1], 1])

        target_pos = self.kin_chain.endpoint_pos(target_state[:self.kin_chain.n_links])
        target_vel = np.zeros(len(target_pos))
        target_state_endpt = np.hstack([target_pos, target_vel, 1])


        current_state = current_state_endpt
        target_state = target_state_endpt
        current_state = np.mat(current_state).reshape(-1,1)
        target_state = np.mat(target_state).reshape(-1,1)

        F = self.F
        A = self.A 
        B = self.B
        Q = self.Q
        R = self.R

        cost = 0
        for k in range(horizon):
            u = F*(target_state - current_state)
            m = current_state - target_state
            cost += (m.T * Q * m + u.T*0*u)[0,0]
            current_state = A*current_state + B*u
        return cost

    def postproc_value(self, values):
        values = np.hstack([-np.inf, values])
        value_diff = values[:-1] - values[1:]
        value_diff[value_diff < 0] = 0
        self._mean = self._mean_alpha*self._mean + (1-self._mean_alpha)*np.mean(value_diff[value_diff > 0])
        value_diff /= self._mean
        return value_diff

    def get_batch(self):
        '''
        see Learner.get_batch for documentation
        '''
        kindata = np.hstack(self.kindata)
        neuraldata = np.hstack(self.neuraldata)
        obs_value = np.hstack(self.obs_value)
        obs_value = self.postproc_value(obs_value)

        self.reset()
        return dict(intended_kin=kindata, spike_counts=neuraldata, value=obs_value)


class CursorGoalLearner2(Learner):
    '''
    CLDA intention estimator based on CursorGoal/Refit-KF ("innovation 1" in Gilja*, Nuyujukian* et al, Nat Neurosci 2012)
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for CursorGoalLearner2

        Parameters
        ----------
        int_speed_type: string, optional, default='dist_to_target'
            Specifies the method to use to estimate the intended speed of the target.
            * dist_to_target: scales based on remaining distance to the target position
            * decoded_speed: use the speed output provided by the decoder, i.e., the difference between the intention and the decoder output can be described by a pure vector rotation

        Returns
        -------
        CursorGoalLearner2 instance
        '''
        int_speed_type = kwargs.pop('int_speed_type', 'dist_to_target')
        self.int_speed_type = int_speed_type
        if not self.int_speed_type in ['dist_to_target', 'decoded_speed']:
            raise ValueError("Unknown type of speed for cursor goal: %s" % self.int_speed_type)

        super(CursorGoalLearner2, self).__init__(*args, **kwargs)

        if self.int_speed_type == 'dist_to_target':
            self.input_state_index = 0

    def calc_int_kin(self, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """
        Calculate the intended kinematics and pair with the neural data
        """
        if state_order is None:
            raise ValueError("New cursor goal requires state order to be specified!")

        # The intended direction (abstract space) from the current state of the decoder to the target state for the task
        int_dir = target_state - decoder_state
        vel_inds, = np.nonzero(state_order == 1)
        pos_inds, = np.nonzero(state_order == 0)
        
        # Calculate intended speed
        if task_state in ['hold', 'origin_hold', 'target_hold']:
            speed = 0
        #elif task_state in ['target', 'origin', 'terminus']:
        else:
            if self.int_speed_type == 'dist_to_target':
                speed = np.linalg.norm(int_dir[pos_inds])
            elif self.int_speed_type == 'decoded_speed':
                speed = np.linalg.norm(decoder_output[vel_inds])
        #else:
        #    speed = np.nan

        int_vel = speed*normalize(int_dir[pos_inds])
        int_kin = np.hstack([decoder_output[pos_inds], int_vel, 1]).reshape(-1, 1)

        if np.any(np.isnan(int_kin)):
            int_kin = None

        return int_kin

    def __call__(self, spike_counts, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """
        Calculate the intended kinematics and pair with the neural data
        """
        if state_order is None:
            raise ValueError("CursorGoalLearner2.__call__ requires state order to be specified!")
        super(CursorGoalLearner2, self).__call__(spike_counts, decoder_state, target_state, decoder_output, task_state, state_order=state_order)
    

###################
## iBMI learners ##
###################

# simple iBMI learners that just use an "assister" object

class ArmAssistLearner(Learner):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        assist_speed   = kwargs.pop('assist_speed', 2.)
        target_radius  = kwargs.pop('target_radius', 2.)
        assister_kwargs = dict(decoder_binlen=decoder_binlen, target_radius=target_radius, assist_speed=assist_speed)
        self.assister = assist.ArmAssistAssister(**assister_kwargs)

        super(ArmAssistLearner, self).__init__(*args, **kwargs)

        self.input_state_index = -1

    def calc_int_kin(self, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """Calculate/estimate the intended ArmAssist kinematics."""
        current_state = decoder_state[:, None]  # assister expects shape to be (7, 1)
        target_state  = target_state[:, None]   # assister expects shape to be (7, 1)
        intended_state = self.assister(current_state, target_state, 1)[0]

        return intended_state

    def __call__(self, neural_features, decoder_state, target_state, decoder_output, task_state, state_order=None):
        '''Calculate the intended kinematics and pair with the neural data.'''
        super(ArmAssistLearner, self).__call__(neural_features, decoder_state, target_state, decoder_output, task_state, state_order=state_order)


class ReHandLearner(Learner):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        assist_speed   = kwargs.pop('assist_speed', 2.)
        target_radius  = kwargs.pop('target_radius', 2.)
        assister_kwargs = dict(decoder_binlen=decoder_binlen, target_radius=target_radius, assist_speed=assist_speed)
        self.assister = assist.ReHandAssister(**assister_kwargs)

        super(ReHandLearner, self).__init__(*args, **kwargs)

        self.input_state_index = -1

    def calc_int_kin(self, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """Calculate/estimate the intended ReHand kinematics."""
        current_state = decoder_state[:, None]  # assister expects shape to be (9, 1)
        target_state  = target_state[:, None]   # assister expects shape to be (9, 1)
        intended_state = self.assister(current_state, target_state, 1)[0]

        return intended_state

    def __call__(self, neural_features, decoder_state, target_state, decoder_output, task_state, state_order=None):
        '''Calculate the intended kinematics and pair with the neural data.'''
        super(ReHandLearner, self).__call__(neural_features, decoder_state, target_state, decoder_output, task_state, state_order=state_order)


class IsMoreLearner(Learner):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        decoder_binlen = kwargs.pop('decoder_binlen', 0.1)
        assist_speed   = kwargs.pop('assist_speed', 2.)
        target_radius  = kwargs.pop('target_radius', 2.)
        assister_kwargs = dict(decoder_binlen=decoder_binlen, target_radius=target_radius, assist_speed=assist_speed)
        self.assister = assist.IsMoreAssister(**assister_kwargs)

        super(IsMoreLearner, self).__init__(*args, **kwargs)

        self.input_state_index = -1

    def calc_int_kin(self, decoder_state, target_state, decoder_output, task_state, state_order=None):
        """Calculate/estimate the intended ArmAssist+ReHand kinematics."""
        current_state = decoder_state[:, None]  # assister expects shape to be (15, 1)
        target_state  = target_state[:, None]   # assister expects shape to be (15, 1)
        intended_state = self.assister(current_state, target_state, 1)[0]

        return intended_state

    def __call__(self, neural_features, decoder_state, target_state, decoder_output, task_state, state_order=None):
        '''Calculate the intended kinematics and pair with the neural data.'''
        super(IsMoreLearner, self).__call__(neural_features, decoder_state, target_state, decoder_output, task_state, state_order=state_order)


# OFC iBMI learners

class ArmAssistOFCLearner(OFCLearner):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, batch_size, *args, **kwargs):
        '''Specific instance of the OFCLearner for the ArmAssist.'''
        dt = kwargs.pop('dt', 0.1)

        ssm = StateSpaceArmAssist()
        A, B, _ = ssm.get_ssm_matrices()
        
        # TODO -- velocity cost? not necessary?
        Q = np.mat(np.diag([1., 1., 1., 0, 0, 0, 0]))
        self.Q = Q
        
        R = 1e7 * np.mat(np.diag([1., 1., 1.]))
        self.R = R

        F = feedback_controllers.LQRController.dlqr(A, B, Q, R)
        F_dict = RegexKeyDict()
        F_dict['.*'] = F

        super(ArmAssistOFCLearner, self).__init__(batch_size, A, B, F_dict, *args, **kwargs)

        self.input_state_index = -1
    
    def calc_int_kin(self, current_state, target_state, decoder_output, task_state, state_order=None):
        '''Overriding to account for proper subtraction of angles.'''
        try:
            current_state = np.mat(current_state).reshape(-1, 1)
            target_state = np.mat(target_state).reshape(-1, 1)
            F = self.F_dict[task_state]
            A = self.A
            B = self.B

            diff = target_state - current_state
            diff[2] = angle_subtract(target_state[2], current_state[2])

            u = F*diff
            state_cost = diff.T * self.Q * diff
            ctrl_cost  = u.T * self.R * u

            # print 'target_state:', target_state
            # print 'state x cost:', diff[0]**2 * float(self.Q[0, 0])
            # print 'state y cost:', diff[1]**2 * float(self.Q[1, 1])
            # print 'state z cost:', diff[2]**2 * float(self.Q[2, 2])
            # print 'u x cost:', u[0]**2 * float(self.R[0, 0])
            # print 'u y cost:', u[1]**2 * float(self.R[1, 1])
            # print 'u z cost:', u[2]**2 * float(self.R[2, 2])
            # print 'state cost:', float(state_cost)
            # print 'ctrl cost:', float(ctrl_cost)
            # print '\n'

            return A*current_state + B*F*(diff)        
        except KeyError:
            return None


class ReHandOFCLearner(OFCLearner):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, batch_size, *args, **kwargs):
        '''Specific instance of the OFCLearner for the ReHand.'''
        dt = kwargs.pop('dt', 0.1)

        ssm = StateSpaceReHand()
        A, B, _ = ssm.get_ssm_matrices()
        
        # TODO -- velocity cost? not necessary?
        Q = np.mat(np.diag([1., 1., 1., 1., 0, 0, 0, 0, 0]))
        self.Q = Q
        
        R = 1e7 * np.mat(np.diag([1., 1., 1., 1.]))
        self.R = R

        F = feedback_controllers.LQRController.dlqr(A, B, Q, R)
        F_dict = RegexKeyDict()
        F_dict['.*'] = F

        super(ReHandOFCLearner, self).__init__(batch_size, A, B, F_dict, *args, **kwargs)

        self.input_state_index = -1
    
    def calc_int_kin(self, current_state, target_state, decoder_output, task_state, state_order=None):
        '''Overriding to account for proper subtraction of angles.'''
        try:
            current_state = np.mat(current_state).reshape(-1, 1)
            target_state = np.mat(target_state).reshape(-1, 1)
            F = self.F_dict[task_state]
            A = self.A
            B = self.B

            diff = target_state - current_state
            for i in range(4):
                diff[i] = angle_subtract(target_state[i], current_state[i])

            return A*current_state + B*F*(diff)        
        except KeyError:
            return None


class IsMoreOFCLearner(OFCLearner):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, batch_size, *args, **kwargs):
        '''Specific instance of the OFCLearner for full IsMore system
        (ArmAssist + ReHand).'''
        dt = kwargs.pop('dt', 0.1)

        ssm = StateSpaceIsMore()
        A, B, _ = ssm.get_ssm_matrices()
        
        # TODO -- velocity cost? not necessary?
        Q = 1*np.mat(np.diag([1., 1., 1., 1., 1., 1., 1., 0, 0, 0, 0, 0, 0, 0, 0]))
        self.Q = Q
        
        R = 1e7 * np.mat(np.diag([1., 1., 1., 1., 1., 1., 1.]))
        self.R = R

        F = feedback_controllers.LQRController.dlqr(A, B, Q, R)
        F_dict = RegexKeyDict()
        F_dict['.*'] = F

        super(IsMoreOFCLearner, self).__init__(batch_size, A, B, F_dict, *args, **kwargs)

        self.input_state_index = -1
    
    def calc_int_kin(self, current_state, target_state, decoder_output, task_state, state_order=None):
        '''Overriding to account for proper subtraction of angles.'''
        try:
            current_state = np.mat(current_state).reshape(-1, 1)
            target_state = np.mat(target_state).reshape(-1, 1)
            F = self.F_dict[task_state]
            A = self.A
            B = self.B

            diff = target_state - current_state
            for i in range(2, 7):
                diff[i] = angle_subtract(target_state[i], current_state[i])

            # print 'diff:'
            # print diff
            # print 'A*current_state'
            # print A*current_state
            BF = B*F
            print 'c1:', BF[7,0]
            print 'c2:', BF[7,7]
            print 'B*F*diff'
            print B*F*diff
            return A*current_state + B*F*(diff)        
        except KeyError:
            return None



##############################################################################
## Updaters
##############################################################################
class Updater(object):
    '''
    Classes for updating decoder parameters
    '''
    def init(self, decoder):
        pass

class CLDARecomputeParameters(mp.Process):
    '''    Docstring    '''
    update_kwargs = dict() 
    def __init__(self, work_queue, result_queue):
        ''' 
        Parameters
        ----------
        work_queue : mp.Queue
            Jobs start when an entry is found in work_queue
        result_queue : mp.Queues
            Results of job are placed back onto result_queue
        '''
        # run base constructor
        super(CLDARecomputeParameters, self).__init__()

        self.work_queue = work_queue
        self.result_queue = result_queue
        self.done = mp.Event()

    def _check_for_job(self):
        '''    Docstring    '''
        try:
            job = self.work_queue.get_nowait()
        except:
            job = None
        return job
        
    def run(self):
        '''    Docstring    '''
        while not self.done.is_set():
            job = self._check_for_job()

            # unpack the data
            if not job == None:
                new_params = self.calc(*job)
                self.result_queue.put(new_params)

            # Pause to lower the process's effective priority
            time.sleep(0.5)

    def calc(self, *args, **kwargs):
        """
        Re-calculate parameters based on input arguments.  This
        method should be overwritten for any useful CLDA to occur!
        """
        return None

    def stop(self):
        '''    Docstring    '''
        self.done.set()

class KFSmoothbatchSingleThread(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def calc(self, intended_kin, spike_counts, decoder, half_life=None, **kwargs):
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

        if half_life is not None:
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

class KFSmoothbatch(KFSmoothbatchSingleThread, CLDARecomputeParameters):
    '''    Docstring    '''
    update_kwargs = dict(steady_state=True)
    def __init__(self, work_queue, result_queue, batch_time, half_life):
        '''    Docstring    '''
        super(KFSmoothbatch, self).__init__(work_queue, result_queue)
        self.half_life = half_life
        self.batch_time = batch_time
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))
        
class KFOrthogonalPlantSmoothbatchSingleThread(KFSmoothbatchSingleThread):
    '''    Docstring    '''
    def __init__(self, default_gain=None):
        self.default_gain = default_gain

    @classmethod
    def scalar_riccati_eq_soln(cls, a, w, n):
        '''    Docstring    '''
        return (1-a*n)/w * (a-n)/n 

    def calc(self, *args, **kwargs):
        '''    Docstring    '''
        # args = (intended_kin, spike_counts, rho, decoder)
        new_params = super(KFOrthogonalPlantSmoothbatchSingleThread, self).calc(*args, **kwargs)
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
                d = KFOrthogonalPlantSmoothbatchSingleThread.scalar_riccati_eq_soln(a, w, n)
                D_diag.append(d)

            D[3:6, 3:6] = np.mat(np.diag(D_diag))

        new_params['kf.C_xpose_Q_inv_C'] = D
        new_params['kf.C_xpose_Q_inv'] = C.T * np.linalg.pinv(Q)
        return new_params


class KFOrthogonalPlantSmoothbatch(KFOrthogonalPlantSmoothbatchSingleThread, KFSmoothbatch):
    '''    Docstring    '''
    def __init__(self, *args, **kwargs):
        '''    Docstring    '''
        self.default_gain = kwargs.pop('default_gain', None)
        KFSmoothbatch.__init__(self, *args, **kwargs)
        
class PPFSmoothbatchSingleThread(object):
    '''    Docstring    '''
    def calc(self, intended_kin, spike_counts, decoder, half_life=None, **kwargs):
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


class PPFSmoothbatch(PPFSmoothbatchSingleThread, CLDARecomputeParameters):
    '''    Docstring    '''
    def __init__(self, work_queue, result_queue, batch_time, half_life):
        '''    Docstring    '''
        super(PPFSmoothbatch, self).__init__(work_queue, result_queue)
        self.half_life = half_life
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))


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
        if intended_kin == None or spike_counts == None or decoder == None:
            raise ValueError("must specify intended_kin, spike_counts and decoder objects for the updater to work!")        
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
            P_params_est_inv = fast_inv(self.P_params_est)
            L = np.dstack([rates[c] * C_xpose_C for c in range(self.n_units)]).transpose([2,0,1])
            self.P_params_est = fast_inv(P_params_est_inv + L)

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
    def __init__(self, work_queue, result_queue, batch_time, half_life):
        '''
        Constructor for KFRML

        Parameters
        ----------
        work_queue : None
            Not used for this method!
        result_queue : None
            Not used for this method!
        batch_time : float
            Size of data batch to use for each update. Specify in seconds.
        half_life : float 
            Amount of time (in seconds) before parameters are half-overwritten by new data.

        Returns
        -------
        KFRML instance

        '''
        self.work_queue = None
        self.batch_time = batch_time
        self.result_queue = None        
        self.half_life = half_life
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))

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
        if values is not None:
            n_samples = np.sum(values)
            B = np.mat(np.diag(values))
        else:
            n_samples = spike_counts.shape[1]
            B = np.mat(np.eye(n_samples))

        self.R = rho*self.R + (x*B*x.T)
        self.S = rho*self.S + (y*B*x.T)
        self.T = rho*self.T + np.dot(y, B*y.T)
        self.ESS = rho*self.ESS + n_samples

        R_inv = np.mat(np.zeros(self.R.shape))
        R_inv[np.ix_(drives_neurons, drives_neurons)] = self.R[np.ix_(drives_neurons, drives_neurons)].I
        C = self.S * R_inv

        Q = (1./self.ESS) * (self.T - self.S*C.T) 

        mFR = (1-rho)*np.mean(spike_counts.T,axis=0) + rho*mFR_old
        sdFR = (1-rho)*np.std(spike_counts.T,axis=0) + rho*sdFR_old

        C_xpose_Q_inv   = C.T * np.linalg.pinv(Q)
        C_xpose_Q_inv_C = C_xpose_Q_inv * C
        
        new_params = {'kf.C':C, 'kf.Q':Q, 
            'kf.C_xpose_Q_inv_C':C_xpose_Q_inv_C, 'kf.C_xpose_Q_inv':C_xpose_Q_inv,
            'mFR':mFR, 'sdFR':sdFR, 'kf.ESS':self.ESS, 'filt.R':self.R, 'filt.S':self.S, 'filt.T':self.T}

        return new_params


class PPFRML(KFRML):
    '''
    Extension of the RML method to the point-process observation model using a Gaussian approximation to the obs model
    '''
    def init(self, decoder):
        n_params_per_cell = decoder.ssm.drives_obs_inds
        n_units = decoder.n_units
        R_inv_cell = np.diag([0.00071857755796282436, 0.00071857755796282436, 0.018144994682778668])
        from scipy.linalg import block_diag
        self.R_inv = block_diag(*([R_inv_cell]*n_units))
        self.S = decoder.filt.S

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
            Half-life to use to calculate the parameter change step size. If not specified, 
            the half-life specified when the Updater was constructed is used.
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

        x = np.mat(intended_kin)
        y = np.mat(spike_counts)
        n_features, n_samples = y.shape

        # if values is not None:
        #     n_samples = np.sum(values)
        #     B = np.mat(np.diag(values))
        # else:
        #     n_samples = spike_counts.shape[1]
        #     B = np.mat(np.eye(n_samples))

        C = decoder.filt.C 
        dt = decoder.filt.dt

        for k in range(n_samples):
            x_t = x[drives_neurons, k]
            Loglambda_predict = C[:,drives_neurons] * x_t
            exp = np.vectorize(lambda x: np.real(cmath.exp(x)))
            lambda_predict = exp(np.array(Loglambda_predict).ravel())/dt
            Q_inv = np.mat(np.diag(lambda_predict*dt))

            y_t = y[:,k]
            # self.R = rho*self.R + np.kron(x_t*x_t.T, Q_inv)
            self.S[:,drives_neurons] = rho*self.S[:,drives_neurons] + Q_inv*y_t*x_t.T

        # self.R = rho*self.R + (x*B*x.T)
        # self.S = rho*self.S + (y*B*x.T)

        # print self.R_inv.shape
        # print self.S[:,drives_neurons].T.flatten().shape
        vec_C = np.dot(self.R_inv, self.S[:,drives_neurons].T.flatten().reshape(-1,1)) #np.linalg.lstsq(self.R, self.S)[0]
        C = np.zeros_like(C)
        C[:,drives_neurons] = vec_C.reshape(-1, n_features).T
        # TODO these aren't necessary for the independent observations PPF, but maybe for the more complicated forms?
        # self.T = rho*self.T + np.dot(y, B*y.T)
        # self.ESS = rho*self.ESS + n_samples
        
        new_params = {'filt.C':C, 'filt.S':self.S} # 'kf.ESS':self.ESS, 'filt.T':self.T

        return new_params


class KFRML_IVC(KFRML):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    default_gain = None
    def calc(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        new_params = super(KFRML_IVC, self).calc(*args, **kwargs)
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
                d = KFOrthogonalPlantSmoothbatchSingleThread.scalar_riccati_eq_soln(a, w, n)
                D_diag.append(d)

            D[3:6, 3:6] = np.mat(np.diag(D_diag))

        new_params['kf.C_xpose_Q_inv_C'] = D
        new_params['kf.C_xpose_Q_inv'] = C.T * np.linalg.pinv(Q)
        return new_params


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
        C[:,-1] = mFR.reshape(-1,1)

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
