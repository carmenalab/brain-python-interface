import multiprocessing as mp
import numpy as np
from riglib.bmi import kfdecoder
import time

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


class CursorGoalLearner(Learner):
    def __init__(self, batch_size, *args, **kwargs):
        super(CursorGoalLearner, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.kindata = []
        self.neuraldata = []
    
    def __call__(self, spike_counts, cursor_pos, target_pos):
        """
        Rotation toward target state
        """

        # estimate intended velocity vector using cursorGoal
        # TODO this needs to be generalized so that the hold
        # the r regular cna be specified simultaneously 
        # cursor_pos = prev_state[0:2]
        int_dir = target_pos - cursor_pos
        int_vel = normalize(int_dir)*0.1
        int_kin = np.hstack([np.zeros(len(int_vel)), int_vel, 1])
        
        if not self.is_full() and self.enabled:
            self.kindata.append(int_kin)
            self.neuraldata.append(spike_counts)
    
    def is_full(self):
        return len(self.kindata) >= self.batch_size

    def get_batch(self):
        kindata = np.vstack(self.kindata).T
        neuraldata = np.vstack(self.neuraldata).T
        self.kindata = []
        self.neuraldata = []
        return kindata, neuraldata


class CLDARecomputeParameters(mp.Process):
    def __init__(self, work_queue, result_queue):
        # run base constructor
        super(CLDARecomputeParameters, self).__init__()

        self.work_queue = work_queue
        self.result_queue = result_queue
        self.done = False

    def _check_for_job(self):
        try:
            job = self.work_queue.get_nowait()
        except:
            job = None
        return job
        
    def run(self):
        while not self.done:
            job = self._check_for_job()

            # unpack the data
            if not job == None:
                new_params = self.calc(*job)
                self.result_queue.put(new_params)

            # Pause to lower the process's effective priority
            time.sleep(0.5)

    def calc(self, *args, **kwargs):
        return None
    def stop(self):
        self.done = True



class KFSmoothbatch(CLDARecomputeParameters):
    def __init__(self, work_queue, result_queue, batch_time, half_life):
        super(KFSmoothbatch, self).__init__(work_queue, result_queue)
        self.hlife = half_life
        self.rho = np.exp(np.log(0.5) / (self.hlife/batch_time))
        
    def calc(self, intended_kin, spike_counts, rho, C_old, Q_old, drives_neurons):
        C_hat, Q_hat = kfdecoder.KalmanFilter.MLE_obs_model(intended_kin, spike_counts, 
            include_offset=False, drives_obs=drives_neurons)
        C = (1-rho)*C_hat + rho*C_old
        Q = (1-rho)*Q_hat + rho*Q_old
        return C, Q

if __name__ == '__main__':
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
