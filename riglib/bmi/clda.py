'''
CLDA classes
'''
import multiprocessing as mp
import numpy as np
from riglib.bmi import kfdecoder, ppfdecoder, train, bmi, feedback_controllers
import time
import cmath
from itertools import izip
import tables


inv = np.linalg.inv

from numpy.linalg import lapack_lite
lapack_routine = lapack_lite.dgesv
# http://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy 
def fast_inv(A):
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

## Learners
def normalize(vec):
    norm_vec = vec / np.linalg.norm(vec)
    
    if np.any(np.isnan(norm_vec)):
        norm_vec = np.zeros(len(vec))
    
    return norm_vec

class Learner(object):
    def __init__(self, *args, **kwargs):
        self.enabled = True
        self.input_state_index = -1

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def reset(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class DumbLearner(Learner):
    def __init__(self, *args, **kwargs):
        self.enabled = False
        self.input_state_index = 0

    def __call__(self, *args, **kwargs):
        """ Do nothing; hence the name of the class"""
        pass

    def is_full(self):
        return False

    def get_batch(self):
        raise NotImplementedError

class BatchLearner(Learner):
    def __init__(self, batch_size, *args, **kwargs):
        super(BatchLearner, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.kindata = []
        self.neuraldata = []
    
    def __call__(self, spike_counts, int_kin):
        """
        Rotation toward target state
        """
        if self.enabled:
            self.kindata.append(int_kin)
            self.neuraldata.append(spike_counts)
    
    def is_full(self):
        return len(self.kindata) >= self.batch_size

    def get_batch(self):
        kindata = np.vstack(self.kindata).T
        neuraldata = np.hstack(self.neuraldata)
        self.kindata = []
        self.neuraldata = []
        return kindata, neuraldata

class OFCLearner(BatchLearner):
    def __init__(self, batch_size, A, B, F_dict, *args, **kwargs):
        super(OFCLearner, self).__init__(batch_size, *args, **kwargs)
        self.B = B
        self.F_dict = F_dict
        self.A = A

    def _target_BMI_state(self, target_pos):
        '''
        Determine the target state of the BMI. Changes depending on the coordinate
        space of the decoder. For now, assume 3D endpoint pos/vel state and
        override in child classes
        '''
        target_state = np.hstack([target_pos.ravel(), np.zeros(len(target_pos)), 1])
        target_state = np.mat(target_state.reshape(-1,1))
        return target_state

    def _run_fbcontroller(self, F, current_state, target_state):
        A = self.A
        B = self.B
        return A*current_state + B*F*(target_state - current_state)

    def __call__(self, spike_counts, cursor_state, target_pos, decoded_vel, task_state):
        if task_state in self.F_dict:
            target_state = self._target_BMI_state(target_pos)
            current_state = np.mat(cursor_state).reshape(-1,1)
            int_state = self._run_fbcontroller(self.F_dict[task_state], current_state, target_state)

            if self.enabled:
                self.kindata.append(int_state)
                self.neuraldata.append(spike_counts)

    def get_batch(self):
        kindata = np.hstack(self.kindata)
        neuraldata = np.hstack(self.neuraldata)
        self.kindata = []
        self.neuraldata = []
        return kindata, neuraldata

class OFCLearner3DEndptPPF(OFCLearner):
    def __init__(self, batch_size, *args, **kwargs):
        '''
        Specific instance of the OFCLearner for the 3D endpoint PPF
        TODO to generalize this better, should be able to store these objects
        to file, just like a decoder, since the feedback matrices may change 
        on different days...
        '''
        dt = kwargs.pop('dt', 1./180)
        use_tau_unNat = kwargs.pop('tau', 2.7)
        self.tau = use_tau_unNat
        print "learner cost fn param: %g" % use_tau_unNat
        #dt = kwargs['dt'] if 'dt' in kwargs else 1./180
        #use_tau_unNat = 1.
        tau_scale = 28*use_tau_unNat/1000
        bin_num_ms = (dt/0.001)
        # 11/6 changed w_r below to 0.3 * ... from 3 * ...
        # 11/6, changed back
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
        
        F = feedback_controllers.dlqr(A, B, Q, R)
        F_dict = dict(target=F, hold=F)
        super(OFCLearner3DEndptPPF, self).__init__(batch_size, A, B, F_dict, *args, **kwargs)

        # Tell AdaptiveBMI that this learner wants the most recent output
        # of the decoder rather than the second most recent, to matcn MATLAB
        self.input_state_index = 0
        
class CursorGoalLearner(Learner):
    def __init__(self, batch_size, *args, **kwargs):
        self.int_speed_type = kwargs.pop('int_speed_type', 'dist_to_target')
        if not self.int_speed_type in ['dist_to_target', 'decoded_speed']:
            raise ValueError("Unknown type of speed for cursor goal: %s" % self.int_speed_type)

        super(CursorGoalLearner, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.kindata = []
        self.neuraldata = []

        if self.int_speed_type == 'dist_to_target':
            self.input_state_index = 0
   
    def __call__(self, spike_counts, cursor_state, target_pos, decoded_vel, 
                 task_state):
        """
        Rotation toward target state
        """
        # estimate intended velocity vector using cursorGoal
        # TODO this needs to be generalized so that the hold
        # the r regular cna be specified simultaneously 
        # cursor_pos = prev_state[0:2]

        cursor_pos = cursor_state[0:len(target_pos)]
        int_dir = target_pos - cursor_pos
        dist_to_targ = np.linalg.norm(int_dir)
        
        # Calculate intended speed
        if self.int_speed_type == 'dist_to_target':
            speed = np.linalg.norm(int_dir)
        elif self.int_speed_type == 'decoded_speed':
            speed = np.linalg.norm(decoded_vel)

        if task_state in ['hold', 'origin_hold', 'target_hold']:
            int_vel = np.zeros(int_dir.shape)            
        elif task_state in ['target', 'origin', 'terminus']:
            int_vel = normalize(int_dir)*speed
        else:
            int_vel = None
        
        if self.enabled and int_vel is not None:
            n_subbins = spike_counts.shape[1]
            int_kin = np.hstack([np.zeros(len(int_vel)), int_vel, 1])
            for k in range(n_subbins):
                self.kindata.append(int_kin)
            self.neuraldata.append(spike_counts)
    
    def is_full(self):
        return len(self.kindata) >= self.batch_size

    def get_batch(self):
        kindata = np.vstack(self.kindata).T
        neuraldata = np.hstack(self.neuraldata)
        self.kindata = []
        self.neuraldata = []
        return kindata, neuraldata

## Updaters
class CLDARecomputeParameters(mp.Process):
    """Generic class for CLDA parameter recomputation"""
    def __init__(self, work_queue, result_queue):
        ''' __init__
        work_queue, result_queue are mp.Queues
        Jobs start when an entry is found in work_queue
        Results of job are placed back onto result_queue
        '''
        # run base constructor
        super(CLDARecomputeParameters, self).__init__()

        self.work_queue = work_queue
        self.result_queue = result_queue
        self.done = mp.Event()

    def _check_for_job(self):
        try:
            job = self.work_queue.get_nowait()
        except:
            job = None
        return job
        
    def run(self):
        """ The main loop """
        while not self.done.is_set():
            job = self._check_for_job()

            # unpack the data
            if not job == None:
                new_params = self.calc(*job)
                self.result_queue.put(new_params)

            # Pause to lower the process's effective priority
            time.sleep(0.5)

    def calc(self, *args, **kwargs):
        """Re-calculate parameters based on input arguments.  This
        method should be overwritten for any useful CLDA to occur!"""
        return None

    def stop(self):
        self.done.set()

class KFSmoothbatchSingleThread(object):
    def calc(self, intended_kin, spike_counts, rho, decoder):
    #def calc(self, intended_kin, spike_counts, rho, C_old, Q_old, drives_neurons, mFR_old, sdFR_old):
        """Smoothbatch calculations

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
        C = (1-rho)*C_hat + rho*C_old
        Q = (1-rho)*Q_hat + rho*Q_old

        mFR = (1-rho)*np.mean(spike_counts.T,axis=0) + rho*mFR_old
        sdFR = (1-rho)*np.std(spike_counts.T,axis=0) + rho*sdFR_old
        #return C, Q, mFR, sdFR
        D = C.T * Q.I * C
        new_params = {'kf.C':C, 'kf.Q':Q, 
            'kf.C_xpose_Q_inv_C':D, 'kf.C_xpose_Q_inv':C.T * Q.I,
            'mFR':mFR, 'sdFR':sdFR}
        return new_params

class KFSmoothbatch(KFSmoothbatchSingleThread, CLDARecomputeParameters):
    def __init__(self, work_queue, result_queue, batch_time, half_life):
        super(KFSmoothbatch, self).__init__(work_queue, result_queue)
        self.half_life = half_life
        self.batch_time = batch_time
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))
        
class KFOrthogonalPlantSmoothbatchSingleThread(KFSmoothbatchSingleThread):
    def __init__(self, default_gain=None):
        self.default_gain = default_gain

    @classmethod
    def scalar_riccati_eq_soln(cls, a, w, n):
        return (1-a*n)/w * (a-n)/n 

    def calc(self, intended_kin, spike_counts, rho, decoder):
        args = (intended_kin, spike_counts, rho, decoder)
        new_params = super(KFOrthogonalPlantSmoothbatchSingleThread, self).calc(*args)
        C, Q, = new_params['kf.C'], new_params['kf.Q']

        D = (C.T * Q.I * C)
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
        new_params['kf.C_xpose_Q_inv'] = C.T * Q.I
        return new_params


class KFOrthogonalPlantSmoothbatch(KFOrthogonalPlantSmoothbatchSingleThread, KFSmoothbatch):
    def __init__(self, *args, **kwargs):
        self.default_gain = kwargs.pop('default_gain', None)
        KFSmoothbatch.__init__(self, *args, **kwargs)
        
class PPFSmoothbatchSingleThread(object):
    def calc(self, intended_kin, spike_counts, rho, decoder):
    #def calc(self, intended_kin, spike_counts, rho, C_old, drives_neurons):
        """
        Smoothbatch calculations

        Run least-squares on (intended_kinematics, spike_counts) to 
        determine the C_hat and Q_hat of new batch. Then combine with 
        old parameters using step-size rho
        """
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
    def __init__(self, work_queue, result_queue, batch_time, half_life):
        super(PPFSmoothbatch, self).__init__(work_queue, result_queue)
        self.half_life = half_life
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))


class PPFContinuousBayesianUpdater(object):
    def __init__(self, decoder, units='cm', param_noise_scale=1.):
        self.n_units = decoder.filt.C.shape[0]
        #self.param_noise_variances = param_noise_variances
        if units == 'm':
            vel_gain = 1e-4
        elif units == 'cm':
            vel_gain = 1e-8

        print "Updater param noise scale %g" % param_noise_scale
        vel_gain *= param_noise_scale
        param_noise_variances = np.array([vel_gain*0.13, vel_gain*0.13, 1e-4*0.06/50])
        self.W = np.tile(np.diag(param_noise_variances), [self.n_units, 1, 1])

        #self.P_params_est_old = np.zeros([self.n_units, 3, 3])
        #for j in range(self.n_units):
        #    self.P_params_est_old[j,:,:] = self.W #Cov_params_init
        #self.P_params_est_old = P_params_est_old
        self.P_params_est = self.W.copy()

        self.neuron_driving_state_inds = np.nonzero(decoder.drives_neurons)[0]
        self.neuron_driving_states = list(np.take(decoder.states, np.nonzero(decoder.drives_neurons)[0]))
        self.n_states = len(decoder.states)
        self.full_size = len(decoder.states)

        self.dt = decoder.filt.dt
        self.beta_est = np.array(decoder.filt.C) #[:,self.neuron_driving_state_inds])

    def calc(self, int_kin_full, spike_obs_full, rho, decoder):
        n_samples = int_kin_full.shape[1]

        # Squash any observed spike counts which are greater than 1
        spike_obs_full[spike_obs_full > 1] = 1
        for k in range(n_samples):
            spike_obs = spike_obs_full[:,k]
            int_kin = int_kin_full[:,k]

            beta_est = self.beta_est[:,self.neuron_driving_state_inds]
            #P_params_est_old = self.P_params_est_old
            int_kin = np.asarray(int_kin).ravel()[self.neuron_driving_state_inds]
            Loglambda_predict = np.dot(int_kin, beta_est.T)
            rates = np.exp(Loglambda_predict)
            if np.any(rates > 1):
                print 'rates > 1!'
                rates[rates > 1] = 1
            unpred_spikes = np.asarray(spike_obs).ravel() - rates

            C_xpose_C = np.outer(int_kin, int_kin)

            #P_params_est = np.zeros([self.n_units, 3, 3]) # TODO remove hardcoding of # of states
            self.P_params_est += self.W
            P_params_est_inv = fast_inv(self.P_params_est)
            L = np.dstack([rates[c] * C_xpose_C for c in range(self.n_units)]).transpose([2,0,1])
            self.P_params_est = fast_inv(P_params_est_inv + L)

            ## for c in range(self.n_units):
            ##     #P_pred = self.P_params_est[c] + self.W[c]
            ##     self.P_params_est[c] = inv(inv(self.P_params_est[c]) + rates[c]*C_xpose_C)

            beta_est += (unpred_spikes * np.dot(int_kin, self.P_params_est).T).T

            # store beta_est
            self.beta_est[:,self.neuron_driving_state_inds] = beta_est

            #self.P_params_est_old = P_params_est

        return {'filt.C': np.mat(self.beta_est.copy())}

class KFContinuousBayesianUpdater(object):
    def __init__(self, decoder, units='cm'):
        self.n_units = decoder.filt.C.shape[0]
        #self.param_noise_variances = param_noise_variances
        if units == 'm':
            vel_gain = 1e-4
        elif units == 'cm':
            vel_gain = 1e-8

        param_noise_variances = np.array([vel_gain*0.13, vel_gain*0.13, 1e-4*0.06/50])
        self.W = np.tile(np.diag(param_noise_variances), [self.n_units, 1, 1])

        self.P_params_est = self.W.copy()

        self.neuron_driving_state_inds = np.nonzero(decoder.drives_neurons)[0]
        self.neuron_driving_states = list(np.take(decoder.states, np.nonzero(decoder.drives_neurons)[0]))
        self.n_states = len(decoder.states)
        self.full_size = len(decoder.states)

        self.dt = decoder.filt.dt
        self.beta_est = np.array(decoder.filt.C) #[:,self.neuron_driving_state_inds])


    def calc(self, int_kin_full, spike_obs_full, rho, decoder):
        n_samples = int_kin_full.shape[1]
        for k in range(n_samples):
            spike_obs = spike_obs_full[:,k]
            int_kin = int_kin_full[:,k]

            beta_est = self.beta_est[:,self.neuron_driving_state_inds]
            int_kin = np.asarray(int_kin).ravel()[self.neuron_driving_state_inds]
            rates = np.dot(int_kin, beta_est.T)
            unpred_spikes = np.asarray(spike_obs).ravel() - rates

            C_xpose_C = np.outer(int_kin, int_kin)

            self.P_params_est += self.W
            P_params_est_inv = fast_inv(self.P_params_est)
            L = np.dstack([rates[c] * C_xpose_C for c in range(self.n_units)]).transpose([2,0,1])
            self.P_params_est = fast_inv(P_params_est_inv + L)

            # Update estimate of beta
            beta_est += (unpred_spikes * np.dot(int_kin, self.P_params_est).T).T
            self.beta_est[:,self.neuron_driving_state_inds] = beta_est

        return {'filt.C': np.mat(self.beta_est.copy())}


class KFRML(object):
    def __init__(self, work_queue, result_queue, batch_time, half_life):
        # super(KFRML, self).__init__(work_queue, result_queue)
        self.work_queue = None
        self.batch_time = batch_time
        self.result_queue = None        
        self.half_life = half_life
        self.rho = np.exp(np.log(0.5) / (self.half_life/batch_time))
        # self.iter_counter = 0

    @staticmethod
    def compute_suff_stats(hidden_state, obs, include_offset=True):
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

        # R = (1./n_pts) * (X * X.T)
        # S = (1./n_pts) * (Y * X.T)
        # T = (1./n_pts) * (Y * Y.T)
        R = (X * X.T)
        S = (Y * X.T)
        T = (Y * Y.T)
        ESS = n_pts  # "effective sample size" (number of points in batch)

        return (R, S, T, ESS)

    def init_suff_stats(self, decoder):
        self.R = decoder.kf.R
        self.S = decoder.kf.S
        self.T = decoder.kf.T
        self.ESS = decoder.kf.ESS

    def calc(self, intended_kin, spike_counts, rho, decoder):
        drives_neurons = decoder.drives_neurons
        mFR_old        = decoder.mFR
        sdFR_old       = decoder.sdFR

        x = intended_kin
        y = spike_counts
        
        # self.R = rho*self.R + (1-rho)*(x*x.T)
        # self.S = rho*self.S + (1-rho)*(y*x.T)
        # self.T = rho*self.T + (1-rho)*(y*y.T)
        # self.iter_counter += 1

        self.R = rho*self.R + (x*x.T)
        self.S = rho*self.S + (y*x.T)
        self.T = rho*self.T + (y*y.T)
        self.ESS = rho*self.ESS + 1

        R_inv = np.mat(np.zeros(self.R.shape))
        R_inv[np.ix_(drives_neurons, drives_neurons)] = self.R[np.ix_(drives_neurons, drives_neurons)].I
        C = self.S * R_inv

        # Q = 1./(1-(rho)**self.iter_counter) * (self.T - self.S*C.T)
        Q = (1./self.ESS) * (self.T - self.S*C.T) 

        mFR = (1-rho)*np.mean(spike_counts.T,axis=0) + rho*mFR_old
        sdFR = (1-rho)*np.std(spike_counts.T,axis=0) + rho*sdFR_old

        C_xpose_Q_inv   = C.T * Q.I
        C_xpose_Q_inv_C = C_xpose_Q_inv * C
        
        new_params = {'kf.C':C, 'kf.Q':Q, 
            'kf.C_xpose_Q_inv_C':C_xpose_Q_inv_C, 'kf.C_xpose_Q_inv':C_xpose_Q_inv,
            'mFR':mFR, 'sdFR':sdFR}

        return new_params

def write_clda_data_to_hdf_table(hdf_fname, data, ignore_none=False):
    '''
    Parameters
    ==========
    hdf_fname : filename of HDF file
    data : list of dictionaries with the same keys and same dtypes for values
    '''
    
    log_file = open('/home/helene/code/bmi3d/log/clda_log', 'w')
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
