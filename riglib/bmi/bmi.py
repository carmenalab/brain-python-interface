'''
High-level classes for BMI, reused between different decoding algorithms
'''
import numpy as np
import traceback
import re
from riglib.plexon import Spikes
import multiprocessing as mp
from itertools import izip

try:
    from plexon import psth
except:
    import warnings
    warnings.warn('psth module not found, using python spike binning function')


def bin_spikes(spikes, units):
    '''
    Python implementation of the psth spike binning function
    '''
    binned_spikes = defaultdict(int)
    for spike in spikes:
        binned_spikes[(spike['chan'], spike['unit'])] += 1
    return np.array([binned_spikes[unit] for unit in units])


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

class GaussianStateHMM():
    def __init__(self, A, W):
        self.A = A
        self.W = W

    def _init_state(self, init_state=None, init_cov=None):
        """
        Initialize the state of the filter with a mean and covariance (uncertainty)
        """
        ## Initialize the BMI state, assuming 
        nS = self.A.shape[0] # number of state variables
        if init_state == None:
            init_state = np.mat( np.zeros([nS, 1]) )
            if self.include_offset: init_state[-1,0] = 1
        if init_cov == None:
            init_cov = np.mat( np.zeros([nS, nS]) )
        self.init_cov = init_cov
        self.state = GaussianState(init_state, init_cov) 
        self.state_noise = GaussianState(0.0, self.W)
        self.obs_noise = GaussianState(0.0, self.Q)


class Decoder(object):
    def get_filter(self):
        raise NotImplementedError

    def update_params(self, new_params):
        for key, val in new_params.items():
            attr_list = key.split('.')
            final_attr = attr_list[-1]
            attr_list = attr_list[:-1]
            attr = self
            while len(attr_list) > 0:
                attr = getattr(self, attr_list[0])
                attr_list = attr_list[1:]
             
            setattr(attr, final_attr, val)
        
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

    def __getitem__(self, idx):
        """
        Get element(s) of the BMI state, indexed by name or number
        """
        alg = self.get_filter()
        if isinstance(idx, int):
            return alg.state.mean[idx, 0]
        elif isinstance(idx, str) or isinstance(idx, unicode):
            idx = self.states.index(idx)
            return alg.state.mean[idx, 0]
        elif np.iterable(idx):
            return np.array([self.__getitem__(k) for k in idx])
        else:
            raise ValueError("KFDecoder: Improper index type: %" % type(idx))

    def __setitem__(self, idx, value):
        """
        Set element(s) of the BMI state, indexed by name or number
        """
        alg = self.get_filter()
        if isinstance(idx, int):
            alg.state.mean[idx, 0] = value
        elif isinstance(idx, str) or isinstance(idx, unicode):
            idx = self.states.index(idx)
            alg.state.mean[idx, 0] = value
        elif np.iterable(idx):
            [self.__setitem__(k, val) for k, val in izip(idx, value)]
        else:
            raise ValueError("KFDecoder: Improper index type: %" % type(idx))

    def __setstate__(self, state):
        """
        Set decoder state after un-pickling
        """
        self.bin_spikes = psth.SpikeBin(state['units'], state['binlen'])
        del state['cells']
        self.__dict__.update(state)
        alg = self.get_filter()
        alg._pickle_init()
        alg._init_state()

        if not self.hasattr('n_subbins'):
            self.n_subbbins = 1

        self.spike_counts = np.zeros([len(state['units']), self.n_subbins])

    def __getstate__(self):
        """Create dictionary describing state of the decoder instance, 
        for pickling"""
        state = dict(cells=self.units)
        exclude = set(['bin_spikes'])
        for k, v in self.__dict__.items():
            if k not in exclude:
                state[k] = v
        return state


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
        ## print 'spike_obs shape', spike_obs.shape
        self.decoder(spike_obs, target=target_pos, assist_inds=pos_inds, **kwargs)
        decoded_state = self.decoder.get_state()

        if (self.decoder.bmicount == self.decoder.bminum - 1):
            self.reset_spike_counts()
        else:
            self.spike_counts += spike_obs
        
        ## if len(spike_obs) == 0: # no timestamps observed
        ##     # TODO spike binning function needs to properly handle not having any timestamps!
        ##     spike_counts = np.zeros((self.decoder.bin_spikes.nunits,))
        ## elif spike_obs.dtype == Spikes.dtype: # Plexnet dtype
        ##     spike_counts = self.decoder.bin_spikes(spike_obs)
        ## else:
        ##     spike_counts = spike_obs

        # send data to learner
        learn_flag = kwargs['learn_flag'] if 'learn_flag' in kwargs else False
        if learn_flag and (self.decoder.bmicount == self.decoder.bminum - 1):
            #print "sending data to learner", self.learner.batch_size, len(self.learner.kindata)
            self.learner(self.spike_counts, prev_state[pos_inds], target_pos, 
                         decoded_state[vel_inds], task_state)

        new_params = None # Default is that no new parameters are available
        update_flag = False

        if self.learner.is_full():
            self.intended_kin, self.spike_counts_batch = self.learner.get_batch()
            rho = self.updater.rho
            drives_neurons = self.decoder.drives_neurons
            clda_data = (self.intended_kin, self.spike_counts_batch, rho, self.decoder.kf.C, self.decoder.kf.Q, drives_neurons, self.decoder.mFR, self.decoder.sdFR)

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
            new_params['spike_counts_batch'] = self.spike_counts_batch
            self.param_hist.append(new_params)
            self.decoder.update_params(new_params)
            self.learner.enable()
            update_flag = True
            ## print "updating params"
            ## print self.decoder.kf.C.shape
            ## print self.decoder.kf.Q.shape

        return decoded_state, update_flag

    def __del__(self):
        # Stop updater process if it's running in a separate process
        if self.mp_updater: 
            self.updater.stop()
