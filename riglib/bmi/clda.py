'''
CLDA classes
'''
import multiprocessing as mp
import numpy as np
from riglib.bmi import kfdecoder, ppfdecoder, train
import time

## Learners
def normalize(vec):
    norm_vec = vec / np.linalg.norm(vec)
    
    if np.any(np.isnan(norm_vec)):
        norm_vec = np.zeros(len(vec))
    
    return norm_vec

class Learner(object):
    def __init__(self, *args, **kwargs):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def reset(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class DumbLearner(Learner):
    def __call__(self, *args, **kwargs):
        """ Do nothing; hence the name of the class"""
        pass

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
        if not self.is_full() and self.enabled:
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


class CursorGoalLearner(Learner):
    def __init__(self, batch_size, *args, **kwargs):
        super(CursorGoalLearner, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.kindata = []
        self.neuraldata = []
    
    def __call__(self, spike_counts, cursor_pos, target_pos, decoded_vel, 
                 task_state):
        """
        Rotation toward target state
        """
        # estimate intended velocity vector using cursorGoal
        # TODO this needs to be generalized so that the hold
        # the r regular cna be specified simultaneously 
        # cursor_pos = prev_state[0:2]
        #print target_pos
        #print cursor_pos
        int_dir = target_pos - cursor_pos
        dist_to_targ = np.linalg.norm(int_dir)
        if task_state in ['hold', 'origin_hold', 'target_hold']:
            int_vel = np.zeros(int_dir.shape)            
        elif task_state in ['target', 'origin', 'terminus']:
            int_vel = normalize(int_dir)*np.linalg.norm(int_dir)
        else:
            int_vel = None
        
        if not self.is_full() and self.enabled and int_vel is not None:
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
        self.hlife = half_life
        self.batch_time = batch_time
        self.rho = np.exp(np.log(0.5) / (self.hlife/batch_time))
        
class KFOrthogonalPlantSmoothbatchSingleThread(KFSmoothbatchSingleThread):
    def calc(self, intended_kin, spike_counts, rho, decoder): #C_old, Q_old, drives_neurons,
#             mFR_old, sdFR_old):
        
        args = (intended_kin, spike_counts, rho, decoder)
        new_params = super(KFOrthogonalPlantSmoothbatchSingleThread, self).calc(*args)
        C, Q, = new_params['kf.C'], new_params['kf.Q']
        D = (C.T * Q.I * C)
        d = np.mean([D[3,3], D[5,5]])
        D[3:6, 3:6] = np.diag([d, d, d])
        #D[2:4, 2:4] = np.mean(np.diag(D)) * np.eye(2) # TODO generalize!
        # TODO calculate the gain from the riccati equation solution (requires A and W)

        new_params['kf.C_xpose_Q_inv_C'] = D
        new_params['kf.C_xpose_Q_inv'] = C.T * Q.I
        return new_params


class KFOrthogonalPlantSmoothbatch(KFOrthogonalPlantSmoothbatchSingleThread, KFSmoothbatch):
    pass


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
        self.hlife = half_life
        self.rho = np.exp(np.log(0.5) / (self.hlife/batch_time))



class KFRML(object):
    def __init__(self, work_queue, result_queue, batch_time, half_life):
        # super(KFRML, self).__init__(work_queue, result_queue)
        self.work_queue = None
        self.batch_time = batch_time
        self.result_queue = None        
        self.hlife = half_life
        self.rho = np.exp(np.log(0.5) / (self.hlife/batch_time))
        self.iter_counter = 0

    @staticmethod
    def compute_suff_stats(hidden_state, obs, include_offset=True):
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
        X = np.mat(X, dtype=np.float64)

        R = (1./T) * (X * X.T)
        S = (1./T) * (Y * X.T)
        T = (1./T) * (Y * Y.T)

        return (R, S, T)

    def init_suff_stats(self, decoder):
        self.R = decoder.kf.R
        self.S = decoder.kf.S
        self.T = decoder.kf.T

    def calc(self, intended_kin, spike_counts, rho, decoder):
        drives_neurons = decoder.drives_neurons
        mFR_old        = decoder.mFR
        sdFR_old       = decoder.sdFR

        x = intended_kin
        y = spike_counts
        
        self.R = rho*self.R + (1-rho)*(x*x.T)
        self.S = rho*self.S + (1-rho)*(y*x.T)
        self.T = rho*self.T + (1-rho)*(y*y.T)
        self.iter_counter += 1

        R_inv = np.mat(np.zeros(self.R.shape))
	R_inv[np.ix_(drives_neurons, drives_neurons)] = self.R[np.ix_(drives_neurons, drives_neurons)].I
        C = self.S * R_inv
        Q = 1./(1-(rho)**self.iter_counter) * (self.T - self.S*C.T)

        mFR = (1-rho)*np.mean(spike_counts.T,axis=0) + rho*mFR_old
        sdFR = (1-rho)*np.std(spike_counts.T,axis=0) + rho*sdFR_old

        C_xpose_Q_inv   = C.T * Q.I
        C_xpose_Q_inv_C = C_xpose_Q_inv * C
        
        new_params = {'kf.C':C, 'kf.Q':Q, 
            'kf.C_xpose_Q_inv_C':C_xpose_Q_inv_C, 'kf.C_xpose_Q_inv':C_xpose_Q_inv,
            'mFR':mFR, 'sdFR':sdFR}

        return new_params


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
