'''
High-level classes for BMI, reused between different decoding algorithms
'''
import numpy as np
import traceback
import re
from riglib.plexon import Spikes
import multiprocessing as mp

class BMI(object):
    '''
    Legacy class that decoders must inherit from for database reasons
    '''
    pass


class GaussianState(object):
    '''
    Class representing a Gaussian state, commonly used to represent the state
    of the BMI in decoders, including the KF and PPF decoders
    '''
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


class AdaptiveBMI(object):
    def __init__(self, decoder, learner, updater):
        self.decoder = decoder
        self.learner = learner
        self.updater = updater
        self.param_hist = []

        self.clda_input_queue = self.updater.work_queue
        self.clda_output_queue = self.updater.result_queue
        self.mp_updater = isinstance(updater, mp.Process)
        if self.mp_updater:
            self.updater.start()
        self.reset_spike_counts()

    def reset_spike_counts(self):
        self.spike_counts = np.zeros([len(self.decoder.units), self.decoder.n_subbins])

    def __call__(self, spike_obs, target_pos, task_state, *args, **kwargs):
        prev_state = self.decoder.get_state()

        dec_state_dim = len(self.decoder.states)
        pos_inds = filter(lambda k: re.match('hand_p', self.decoder.states[k]), range(dec_state_dim))
        vel_inds = filter(lambda k: re.match('hand_v', self.decoder.states[k]), range(dec_state_dim))

        # run the decoder
        self.decoder(spike_obs, target=target_pos, assist_inds=pos_inds, **kwargs)
        decoded_state = self.decoder.get_state()

        if (self.decoder.bmicount == self.decoder.bminum - 1):
            self.reset_spike_counts()
        else:
            self.spike_counts += spike_obs
        
        if len(spike_obs) == 0: # no timestamps observed
            # TODO spike binning function needs to properly handle not having any timestamps!
            spike_counts = np.zeros((self.decoder.bin_spikes.nunits,))
        elif spike_obs.dtype == Spikes.dtype: # Plexnet dtype
            spike_counts = self.decoder.bin_spikes(spike_obs)
        else:
            spike_counts = spike_obs

        learn_flag = kwargs['learn_flag'] if 'learn_flag' in kwargs else False
        #print self.decoder.bmicount, self.decoder.bminum-1
        if learn_flag and (self.decoder.bmicount == self.decoder.bminum - 1):
            self.learner(spike_counts, prev_state[pos_inds], target_pos, 
                         decoded_state[vel_inds], task_state)

        # send data to learner
        new_params = None # Default is that now new parameters are available
        update_flag = False

        if self.learner.is_full():
            self.intended_kin, self.spike_counts = self.learner.get_batch()
            rho = self.updater.rho
            drives_neurons = self.decoder.drives_neurons
            clda_data = (self.intended_kin, self.spike_counts, rho, self.decoder.kf.C, self.decoder.kf.Q, drives_neurons, self.decoder.mFR, self.decoder.sdFR)

            if self.mp_updater:
                self.clda_input_queue.put(clda_data)
                # Deactivate learner until parameter update is received
                self.learner.disable() 
            else:
                new_params = self.updater.calc(*clda_data)
                #self.decoder.update_params(new_params)

        if self.mp_updater:
            # Check for a parameter update
            try:
                new_params = self.clda_output_queue.get_nowait()
            except:
                import os
                homedir = os.getenv('HOME')
                logfile = os.path.join(homedir, 'Desktop/clda_log')
                f = open(logfile, 'w')
                traceback.print_exc(file=f)
                f.close()

        if new_params is not None:
            new_params['intended_kin'] = self.intended_kin
            new_params['spike_counts'] = self.spike_counts
            self.param_hist.append(new_params)
            self.decoder.update_params(new_params)
            self.learner.enable()
            update_flag = True

        return decoded_state, update_flag

    def __del__(self):
        # Stop updater process if it's running in a separate process
        if self.mp_updater: 
            self.updater.stop()
