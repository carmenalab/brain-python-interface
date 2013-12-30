'''
High-level classes for BMI, reused between different decoding algorithms
'''
import numpy as np
import traceback
import re
from riglib.plexon import Spikes
import multiprocessing as mp
from itertools import izip
import time
import re

gen_joint_coord_regex = re.compile('.*?_p.*')

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
        if isinstance(mean, np.matrix):
            assert mean.shape[1] == 1 # column vector
            self.mean = mean
        elif isinstance(mean, float):
            self.mean = mean
        elif isinstance(mean, np.ndarray):
            if np.ndim(mean) == 1:
                mean = mean.reshape(-1,1)
            self.mean = np.mat(mean)

        # Covariance
        assert cov.shape[0] == cov.shape[1] # Square matrix
        if isinstance(cov, np.ndarray):
            cov = np.mat(cov)
        self.cov = cov
    
    def __rmul__(self, other):
        if isinstance(other, np.matrix):
            mu = other*self.mean
            cov = other*self.cov*other.T
        elif isinstance(other, int) or isinstance(other, np.float64) or isinstance(other, float):
            mu = other*self.mean
            cov = other**2 * self.cov
        elif isinstance(other, np.ndarray):
            other = np.mat(other)
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
            return GaussianState(self.mean+other.mean, self.cov+other.cov)
        elif isinstance(other, np.matrix) and other.shape == self.mean.shape:
            return GaussianState(self.mean + other, self.cov)
        else:
            raise ValueError("Gaussian state: cannot add type :%s" % type(other))

class GaussianStateHMM():
    model_attrs = []
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

    def _ssm_pred(self, state, target_state=None):
        A = self.A
        if target_state == None:
            return A*state + self.state_noise
        else:
            B = self.B
            F = self.F
            return (A - B*F)*state + B*F*target_state + self.state_noise

    def __eq__(self, other):
        import train
        return train.obj_eq(self, other, self.model_attrs)

    def __sub__(self, other):
        import train
        return train.obj_diff(self, other, self.model_attrs)


class Decoder(object):
    clda_dtype = [] # define parameters to store in HDF file during CLDA
    def get_filter(self):
        raise NotImplementedError

    def save_params_to_hdf(self, task_data):
        pass

    def plot_pds(self, C, ax=None, plot_states=['hand_vx', 'hand_vz'], **kwargs):
        import matplotlib.pyplot as plt
        if ax == None:
            plt.figure()
            ax = plt.subplot(111)
            ax.hold(True)

        state_inds = [self.states.index(x) for x in plot_states]
        x, z = state_inds
        n_neurons = C.shape[0]
        linestyles = ['-.', '-', '--', ':']
        for k in range(n_neurons):
            unit_str = '%d%s' % (self.units[k,0], chr(96 + self.units[k,1]))
            ax.plot([0, C[k, x]], [0, C[k, z]], label=unit_str, linestyle=linestyles[k/7 % len(linestyles)], **kwargs)
        ax.legend(bbox_to_anchor=(1.1, 1.05), prop=dict(size=8))
        ax.set_title(self)

    def plot_C(self, **kwargs):
        self.plot_pds(self.filt.C, linewidth=2, **kwargs)

    def update_params(self, new_params, **kwargs):
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

        Warning: The variable 'q' is a reserved keyword, referring to all of
        the position states. This strange letter choice was made to be consistent
        with the robotics literature, where 'q' refers to the vector of 
        generalized joint coordinates.
        """
        if isinstance(idx, int):
            return self.filt.state.mean[idx, 0]
        elif idx == 'q':
            pos_states = filter(gen_joint_coord_regex.match, self.states)
            return np.array([self.__getitem__(k) for k in pos_states])
        elif isinstance(idx, str) or isinstance(idx, unicode):
            idx = self.states.index(idx)
            return self.filt.state.mean[idx, 0]
        elif np.iterable(idx):
            return np.array([self.__getitem__(k) for k in idx])
        else:
            raise ValueError("Decoder: Improper index type: %s" % type(idx))

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

    def bin_spikes(self, spikes, max_units_per_channel=13):
        '''
        Count up the number of BMI spikes in a list of spike timestamps
        '''
        unit_inds = self.units[:,0]*max_units_per_channel + self.units[:,1]
        edges = np.sort(np.hstack([unit_inds - 0.5, unit_inds + 0.5]))
        spiking_unit_inds = spikes['chan']*max_units_per_channel + spikes['unit']
        counts, _ = np.histogram(spiking_unit_inds, edges)
        return counts[::2]

    def __setstate__(self, state):
        """
        Set decoder state after un-pickling
        """
        self.__dict__.update(state)
        alg = self.get_filter()
        alg._pickle_init()
        alg._init_state()

        if not hasattr(self, 'n_subbins'):
            self.n_subbins = 1

        if not hasattr(self, 'interpolate_using_ssm'):
            self.interpolate_using_ssm = False

        if not hasattr(self, 'bmicount'):
            self.bmicount = 0

        self.spike_counts = np.zeros([len(state['units']), self.n_subbins])
        self._pickle_init()

    def _pickle_init(self):
        pass

    def __getstate__(self):
        """Create dictionary describing state of the decoder instance, 
        for pickling"""
        state = dict(cells=self.units)
        exclude = set(['bin_spikes'])
        for k, v in self.__dict__.items():
            if k not in exclude:
                state[k] = v
        return state

    def get_state(self):
        '''
        Get the state of the decoder (mean of the Gaussian RV representing the
        state of the BMI)
        '''
        #alg = self.get_filter()
        return np.array(self.filt.state.mean).ravel()

    def predict(self, spike_counts, target=None, speed=0.5, target_radius=2,
                assist_level=0.0, assist_inds=[0,1,2],
                **kwargs):
        """Decode the spikes"""
        # Save the previous cursor state for assist
        prev_kin = self.filt.get_mean()
        if assist_level > 0:
            cursor_pos = prev_kin[assist_inds]
            diff_vec = target - cursor_pos 
            dist_to_target = np.linalg.norm(diff_vec)
            dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
            
            if dist_to_target > target_radius:
                assist_cursor_pos = cursor_pos + speed*dir_to_target
            else:
                assist_cursor_pos = cursor_pos + speed*diff_vec/2

            assist_cursor_vel = (assist_cursor_pos-cursor_pos)/self.binlen
            assist_cursor_kin = np.hstack([assist_cursor_pos, assist_cursor_vel, 1])

        # TODO put this back in for the KF
        ### # re-normalize the variance of the spike observations, if nec
        ### if self.zscore:
        ###     spike_counts = (spike_counts - self.mFR_diff) * self.sdFR_ratio
        ###     # set the spike count of any unit with a 0 mean to it's original mean
        ###     # This is essentially removing it from the decoder.
        ###     spike_counts[self.zeromeanunits] = self.mFR[self.zeromeanunits] 

        # re-format as a 1D col vec
        spike_counts = np.mat(spike_counts.reshape(-1,1))

        # Run the filter
        self.filt(spike_counts)

        # Bound cursor, if any hard bounds for states are applied
        # self.bound_state()

        # if assist_level > 0:
        #     cursor_kin = self.filt.get_mean()
        #     kin = assist_level*assist_cursor_kin + (1-assist_level)*cursor_kin
        #     self.filt.state.mean[:,0] = kin.reshape(-1,1)
        #     self.bound_state()

        state = self.filt.get_mean()
        return state

    def __str__(self):
        if hasattr(self, 'db_entry'):
            return self.db_entry.name
        else:
            return super(Decoder, self).__str__()

    @property
    def n_states(self):
        return len(self.states)

    @property
    def n_units(self):
        '''
        Return the number of units used in the decoder; For both the PPF and
        the KF, this corresponds to the first dimension of the C matrix, 
        but future decoders may need to return a different quantity
        '''
        return self.filt.C.shape[0]

    def __call__(self, obs_t, **kwargs):
        decoding_rate = 1./self.binlen
        if decoding_rate >= 60:
            # Infer the number of sub-bins from the size of the spike counts mat to decode
            n_subbins = obs_t.shape[1]

            outputs = []
            for k in range(n_subbins):
                outputs.append(self.predict(obs_t[:,k], **kwargs))

            return np.vstack(outputs).T
        elif decoding_rate < 60:
            self.spike_counts += obs_t.reshape(-1, 1)
            if self.bmicount == self.bminum-1:
                # Update using spike counts
                self.bmicount = 0
                #print "old = ", np.around(self.filt.get_mean(), decimals=2)
                self.predict(self.spike_counts, **kwargs)
                #print "new = ", np.around(self.filt.get_mean(), decimals=2)
                self.spike_counts = np.zeros([len(self.units), 1])
            else:
                self.bmicount += 1
            return self.filt.get_mean().reshape(-1,1)


class AdaptiveBMI(object):
    def __init__(self, decoder, learner, updater):
        self.decoder = decoder 
        self.learner = learner
        self.updater = updater
        self.param_hist = []

        # Establish inter-process communication mechanisms if the updater runs
        # in a separate process
        self.mp_updater = isinstance(updater, mp.Process)
        if self.mp_updater:
            self.clda_input_queue = self.updater.work_queue
            self.clda_output_queue = self.updater.result_queue
            self.updater.start()
        self.reset_spike_counts()

    def reset_spike_counts(self):
        self.spike_counts = np.zeros([len(self.decoder.units), 1])
        #self.spike_counts = np.zeros([len(self.decoder.units), self.decoder.n_subbins])

    def __call__(self, spike_obs, target_pos, task_state, *args, **kwargs):
        n_units, n_obs = spike_obs.shape

        # If the target is specified as a 1D position (default behavior for 
        # Python BMI tasks), tile to match the number of dimensions as the 
        # spike counts
        if np.ndim(target_pos) == 1:
            target_pos = np.tile(target_pos, [n_obs, 1]).T
            
        dec_state_dim = len(self.decoder.states)
        pos_inds = filter(lambda k: re.match('hand_p', self.decoder.states[k]), range(dec_state_dim))
        vel_inds = filter(lambda k: re.match('hand_v', self.decoder.states[k]), range(dec_state_dim))

        decoded_states = []
        update_flag = False
        learn_flag = kwargs.pop('learn_flag', False)

        for k in range(n_obs):
            spike_obs_k = spike_obs[:,k].reshape(-1,1)
            target_pos_k = target_pos[:,k]

            # NOTE: the conditional below should only ever be active when trying
            # to run this code on MATLAB data! In all python cases, the task_state
            # should accurately reflect the validity of the presented target position
            if np.any(np.isnan(target_pos_k)):
                task_state = 'no_target' 

            # run the decoder
            prev_state = self.decoder.get_state()
            self.decoder(spike_obs_k, target=target_pos_k, assist_inds=pos_inds, **kwargs)
            decoded_state = self.decoder.get_state()
            decoded_states.append(decoded_state)

            # Determine whether the current state or previous state should be given to the learner
            if self.learner.input_state_index == 0:
                learner_state = decoded_state
            elif self.learner.input_state_index == -1:
                learner_state = prev_state
            else:
                print "Not implemented yet: %d" % self.learner.input_state_index
                learner_state = prev_state

            self.spike_counts += spike_obs_k
            if learn_flag and self.decoder.bmicount == 0:
                self.learner(self.spike_counts.copy(), learner_state, target_pos_k, 
                             decoded_state[vel_inds], task_state)
                self.reset_spike_counts()
            elif self.decoder.bmicount == 0:
                self.reset_spike_counts()
        
            new_params = None # by default, no new parameters are available

            if self.learner.is_full():
                self.intended_kin, self.spike_counts_batch = self.learner.get_batch()
                if 'half_life' in kwargs and hasattr(self.updater, 'half_life'):
                    half_life = kwargs['half_life']
                    rho = np.exp(np.log(0.5)/(half_life/self.updater.batch_time))
                elif hasattr(self.updater, 'half_life'):
                    rho = self.updater.rho
                else:
                    rho = -1
                clda_data = (self.intended_kin, self.spike_counts_batch, rho, self.decoder)

                if self.mp_updater:
                    self.clda_input_queue.put(clda_data)
                    # Disable learner until parameter update is received
                    self.learner.disable() 
                else:
                    new_params = self.updater.calc(*clda_data)

            # If the updater is running in a separate process, check if a new 
            # parameter update is available
            if self.mp_updater:
                try:
                    new_params = self.clda_output_queue.get_nowait()
                except:
                    import os
                    homedir = os.getenv('HOME')
                    logfile = os.path.join(homedir, 'Desktop/clda_log')
                    f = open(logfile, 'w')
                    traceback.print_exc(file=f)
                    f.close()

            # Update the decoder if new parameters are available
            if new_params is not None:
                new_params['intended_kin'] = self.intended_kin
                new_params['spike_counts_batch'] = self.spike_counts_batch
                self.param_hist.append(new_params)
                import clda
                if isinstance(self.updater, clda.KFRML):
                    steady_state = False
                else:
                    steady_state = True
                self.decoder.update_params(new_params, steady_state=steady_state)
                self.learner.enable()
                update_flag = True
            else:
                self.param_hist.append(None)


        decoded_states = np.vstack(decoded_states).T
        return decoded_states, update_flag

    def __del__(self):
        # Stop updater if it's running in a separate process
        if self.mp_updater: 
            self.updater.stop()
