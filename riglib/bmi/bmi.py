'''
High-level classes for BMI used to tie all th BMI subcomponent systems together
'''
import numpy as np
import traceback
import re
from riglib.plexon import Spikes
import multiprocessing as mp
import Queue
from itertools import izip
import time
import re
import os
import tables

gen_joint_coord_regex = re.compile('.*?_p.*')


class GaussianState(object):
    '''
    Class representing a multivariate Gaussian. Gaussians are 
    commonly used to represent the state
    of the BMI in decoders, including the KF and PPF decoders
    '''
    def __init__(self, mean, cov):
        '''
        Parameters
        mean: np.array of shape (N, 1) or (N,)
            N-dimensional vector representing the mean of the multivariate Gaussian distribution
        cov: np.array of shape (N, N)
            N-dimensional covariance matrix
        '''
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
        '''
        Gaussian RV multiplication: 
        If X ~ N(mu, sigma) and A is a matrix, then A*X ~ N(A*mu, A*sigma*A.T)
        '''
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
        '''
        Gaussian RV multiplication: 
        If X ~ N(mu, sigma) and A is a matrix, then A*X ~ N(A*mu, A*sigma*A.T)
        '''        
        mean = other*self.mean
        if isinstance(other, int) or isinstance(other, np.float64) or isinstance(other, float):
            cov = other**2 * self.cov
        else:
            print type(other)
            raise
        return GaussianState(mean, cov)

    def __add__(self, other):
        '''
        Gaussian RV addition: If X ~ N(mu1, sigma1) and Y ~ N(mu2, sigma2), then
        X + Y ~ N(mu1 + mu2, sigma1 + sigma2). If Y is a scalar, then X + Y ~ N(mu1 Y, sigma1)
        '''
        if isinstance(other, GaussianState):
            return GaussianState(self.mean+other.mean, self.cov+other.cov)
        elif isinstance(other, np.matrix) and other.shape == self.mean.shape:
            return GaussianState(self.mean + other, self.cov)
        else:
            raise ValueError("Gaussian state: cannot add type :%s" % type(other))


class GaussianStateHMM(object):
    '''
    General hidden Markov model decoder where the state is represented as a Gaussian random vector
    '''
    model_attrs = []
    def __init__(self, A, W):
        '''
        Constructor for GaussianStateHMM

        x_{t+1} = A*x_t + w_t; w_t ~ N(0, W)

        Parameters
        ----------
        A: np.mat of shape (N, N)
            State transition matrix
        W: np.mat of shape (N, N)
            Noise covariance
        '''
        self.A = A
        self.W = W

    def get_mean(self):
        '''
        Return just the mean of the Gaussian representing the state estimate as a 1D array
        '''
        return np.array(self.state.mean).ravel()      

    def _init_state(self, init_state=None, init_cov=None):
        """
        Initialize the state of the filter with a mean and covariance (uncertainty)

        Parameters
        ----------
        init_state : np.matrix, optional
            Initial estimate of the unknown state. If unspecified, a vector of all 0's 
            will be used (except for the offset state, if one exists).
        init_cov : np.matrix, optional
            Uncertainty about the initial state. If unspecified, it is assumed that there
            is no uncertainty (a matrix of all 0's).
        
        Returns
        -------
        None
        """
        ## Initialize the BMI state, assuming 
        nS = self.A.shape[0] # number of state variables
        if init_state == None:
            init_state = np.mat( np.zeros([nS, 1]) )
            if self.include_offset: init_state[-1,0] = 1
        if init_cov == None:
            init_cov = np.mat( np.zeros([nS, nS]) )
        self.state = GaussianState(init_state, init_cov) 
        self.init_noise_models()

    def init_noise_models(self):
        '''
        Initialize the process and observation noise models. The state noise should be 
        Gaussian (as implied by the name of this class). The observation noise may be 
        non-Gaussian depending on the observation model.
        '''
        self.state_noise = GaussianState(0.0, self.W)
        self.obs_noise = GaussianState(0.0, self.Q)

    def _ssm_pred(self, state, u=None, Bu=None, target_state=None, F=None):
        ''' Docstring
        Run the "predict" step of the Kalman filter/HMM inference algorithm:
            x_{t+1|t} = N(Ax_{t|t}, AP_{t|t}A.T + W)

        Parameters
        ----------
        state: GaussianState instance
            State estimate and estimator covariance of current state
        u: np.mat 
        

        Returns
        -------
        GaussianState instance
            Represents the mean and estimator covariance of the new state estimate
        '''
        A = self.A

        if Bu is not None:
            return A*state + Bu + self.state_noise
        elif u is not None:
            Bu = self.B * u
            return A*state + Bu + self.state_noise
        elif target_state is not None:
            B = self.B
            if F == None:
                F = self.F
            return (A - B*F)*state + B*F*target_state + self.state_noise
        else:
            return A*state + self.state_noise

    def __eq__(self, other):
        '''
        Determine equality of two GaussianStateHMM instances
        '''
        import train
        return train.obj_eq(self, other, self.model_attrs)

    def __sub__(self, other):
        '''
        Subtract the model attributes of two GaussianStateHMM instances. Used to determine approximate equality, i.e., equality modulo floating point error
        '''
        import train
        return train.obj_diff(self, other, self.model_attrs)

    def __call__(self, obs, **kwargs):
        """
        When the object is called directly, it's a wrapper for the 
        1-step forward inference function.
        """
        self.state = self._forward_infer(self.state, obs, **kwargs)
        return self.state.mean

    def _pickle_init(self):
        pass

    def __setstate__(self, state):
        """
        Unpickle decoders by loading all the saved parameters and then running _pickle_init

        Parameters
        ----------
        state : dict
            Provided by the unpickling system

        Returns
        -------
        None
        """
        self.__dict__ = state
        self._pickle_init()


class Decoder(object):
    '''
    All BMI decoders should inherit from this class
    '''
    def __init__(self, filt, units, ssm, binlen=0.1, n_subbins=1, tslice=[-1,-1], call_rate=60.0, **kwargs):
        """ 
        Parameters
        ----------
        filt : PointProcessFilter or KalmanFilter instance
            Generic inference algorithm that does the actual observation decoding
        units : array-like
            N x 2 array of units, where each row is (chan, unit)
        ssm : state_space_models.StateSpace instance 
            The state-space model describes the states tracked by the decoder, whether or not
            they are stochastic/related to the observations, bounds on the state, etc.
        binlen : float, optional, default = 0.1
            Bin-length specified in seconds. Gets rounded to a multiple of 1./60
            to match the update rate of the task
        n_subbins : int, optional, default = 3
            Neural observations are always acquired at the 60Hz screen update rate.
            This parameter explains how many bins to sub-divide the observations 
            into. Default of 3 is intended to correspond to ~180Hz / 5.5ms bins
        tslice : array_like, optional, default=[-1, -1]
            start and end times for the neural data used to train, e.g. from the .plx file
        call_rate: float, optional, default = 60 Hz
            Rate in Hz at which the task will run the __call__ function.
        """

        self.filt = filt
        self.filt._init_state()
        self.ssm = ssm

        self.units = np.array(units, dtype=np.int32)
        self.binlen = binlen
        self.bounding_box = ssm.bounding_box
        self.states = ssm.state_names
        
        # The tslice parameter below properly belongs in the database and
        # not in the decoder object because the Decoder object has no record of 
        # which plx file it was trained from. This is a leftover from when it
        # was assumed that every decoder would be trained entirely from a plx
        # file (i.e. and not CLDA)
        self.tslice = tslice
        self.states_to_bound = ssm.states_to_bound
        
        self.drives_neurons = ssm.drives_obs #drives_neurons
        self.n_subbins = n_subbins

        self.bmicount = 0
        self.bminum = int(self.binlen/(1/call_rate))
        self.spike_counts = np.zeros([len(units), 1])

        self.set_call_rate(call_rate)

        self._pickle_init()

    def _pickle_init(self):
        '''
        Functionality common to unpickling a Decoder from file and instantiating a new Decoder.
        A call to this function is the last line in __init__ as well as __setstate__.
        '''
        import train

        # If the decoder doesn't have an 'ssm' attribute, then it's an old
        # decoder in which case the ssm is the 2D endpoint SSM
        if not hasattr(self, 'ssm'):
            self.ssm = train.endpt_2D_state_space

        # Assign a default call rate of 60 Hz and initialize the bmicount/bminum attributes
        if hasattr(self, 'call_rate'):
            self.set_call_rate(self.call_rate)
        else:
            self.set_call_rate(60.0)

    def plot_pds(self, C, ax=None, plot_states=['hand_vx', 'hand_vz'], invert=False, **kwargs):
        '''
        Plot 2D "preferred directions" of features in the Decoder

        Parameters
        ----------
        C: np.array of shape (n_features, n_states)
        ax: matplotlib.pyplot axis, default=None
            axis to plot on. If None specified, a new one is created. 
        plot_states: list of strings, default=['hand_vx', 'hand_vz']
            List of decoder states to plot. Only two can be specified currently
        invert: bool, default=False
            If true, flip the signs of the arrows plotted
        kwargs: dict
            Keyword arguments for the low-level matplotlib function
        '''
        import matplotlib.pyplot as plt
        if ax == None:
            plt.figure()
            ax = plt.subplot(111)
            ax.hold(True)

        if C.shape[1] > 2:
            state_inds = [self.states.index(x) for x in plot_states]
            x, z = state_inds
        else:
            x, z = 0, 1
        n_neurons = C.shape[0]
        linestyles = ['-.', '-', '--', ':']
        if invert:
            C = C*-1
        for k in range(n_neurons):
            unit_str = '%d%s' % (self.units[k,0], chr(96 + self.units[k,1]))
            ax.plot([0, C[k, x]], [0, C[k, z]], label=unit_str, linestyle=linestyles[k/7 % len(linestyles)], **kwargs)
        ax.legend(bbox_to_anchor=(1.1, 1.05), prop=dict(size=8))
        try:
            ax.set_xlabel(plot_states[0])
            ax.set_ylabel(plot_states[1])
        except:
            pass
        ax.set_title(self)

    def plot_C(self, **kwargs):
        '''
        Plot the C matrix (see plot_pds docstring), which is used
        by the KFDecoder and the PPFDecoder
        '''
        self.plot_pds(self.filt.C, linewidth=2, **kwargs)

    def update_params(self, new_params, **kwargs):
        '''
        Method for updating the parameters of the decoder

        Parameters
        ----------
        new_params: dict 
            Keys are the parameters to be replaced, values are the new value of 
            the parameter to replace. In particular, the keys can be dot-separated,
            e.g. to set the attribute 'self.kf.C', the key would be 'kf.C'
        '''
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
        """
        Apply bounds on state vector, if bounding box is specified
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

        Parameters
        ----------
        idx: int or string
            Name of the state, index of the state, or list of indices/names 
            of the Decoder state(s) to return
        """
        if isinstance(idx, int):
            return self.filt.state.mean[idx, 0]
        elif idx == 'q':
            pos_states, = np.nonzero(self.ssm.state_order == 0)
            return np.array([self.__getitem__(k) for k in pos_states])
        elif idx == 'qdot':
            vel_states, = np.nonzero(self.ssm.state_order == 1)
            return np.array([self.__getitem__(k) for k in vel_states])      
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

        Parameters
        ----------
        idx: int or string
            Name of the state, index of the state, or list of indices/names 
            of the Decoder state(s) to return
        """
        if isinstance(idx, int):
            self.filt.state.mean[idx, 0] = value
        elif idx == 'q':
            pos_states, = np.nonzero(self.ssm.state_order == 0)
            self.filt.state.mean[pos_states, 0] = value
        elif idx == 'qdot':
            vel_states, = np.nonzero(self.ssm.state_order == 1)
            self.filt.state.mean[vel_states, 0] = value
        elif idx == 'q_stoch':
            pos_states = filter(lambda k: gen_joint_coord_regex.match(self.states[k]) and self.states[k].stochastic, range(len(self.states)))
            self.filt.state.mean[pos_states, 0] = value
        elif isinstance(idx, str) or isinstance(idx, unicode):
            idx = self.states.index(idx)
            self.filt.state.mean[idx, 0] = value
        elif np.iterable(idx):
            [self.__setitem__(k, val) for k, val in izip(idx, value)]
        else:
            raise ValueError("KFDecoder: Improper index type: %" % type(idx))

    def __setstate__(self, state):
        """
        Set decoder state after un-pickling
        """
        if 'db_entry' in state:
            del state['db_entry']
        self.__dict__.update(state)
        self.filt._pickle_init()
        self.filt._init_state()

        if not hasattr(self, 'n_subbins'):
            self.n_subbins = 1

        if not hasattr(self, 'interpolate_using_ssm'):
            self.interpolate_using_ssm = False

        if not hasattr(self, 'bmicount'):
            self.bmicount = 0

        # if not hasattr(self, 'n_features'):
        #     self.n_features = len(self.units)
        self.n_features = len(self.units)            

        # self.spike_counts = np.zeros([len(state['units']), self.n_subbins])
        self.spike_counts = np.zeros([self.n_features, self.n_subbins])
        self._pickle_init()

    def __getstate__(self):
        """
        Create dictionary describing state of the decoder instance, 
        for pickling.

        NOTE: this function was used originally to avoid pickling a 'bin_spikes',
        which was a function added to the Decoder as an attribute rather than
        being declared as a method. It should no longer be necessary. 
        """
        state = dict(cells=self.units)
        exclude = set(['bin_spikes'])
        for k, v in self.__dict__.items():
            if k not in exclude:
                state[k] = v
        return state

    def set_call_rate(self, call_rate):
        '''
        Function for the higher-level task to set the frequency of function calls to __call__

        Parameters
        ----------
        call_rate : float 
            1./call_rate should be an integer multiple or divisor of the Decoder's 'binlen'

        Returns
        -------
        None
        '''
        self.call_rate = call_rate
        self.bmicount = 0
        self.bminum = int(self.binlen/(1./self.call_rate))
        self.n_subbins = int(np.ceil(1./self.binlen /self.call_rate))

    def get_state(self, shape=-1):
        '''
        Get the state of the decoder (mean of the Gaussian RV representing the
        state of the BMI)
        '''
        return np.asarray(self.filt.state.mean).reshape(shape)

    def predict(self, neural_obs, assist_level=0.0, Bu=None, x_target=None, F=None, weighted_avg_lfc=False, **kwargs):
        """
        Decode the spikes

        Parameters
        ----------
        neural_obs: np.array of shape (N,) or (N, 1)
            One time-point worth of neural features to decode
        assist_level: float
            Weight given to the assist term
        Bu: np.mat of shape (N, 1)
            Assist vector to be added on to the Decoder state. Must be of the same dimension 
            as the state vector.
        kwargs: dict
            Mostly for kwargs function call compatibility
        """
        if assist_level > 0 and Bu == None:
            raise ValueError("Assist cannot be used if the forcing term is not specified!")

        # re-normalize the variance of the spike observations, if nec
        if hasattr(self, 'zscore') and self.zscore:
            neural_obs = (np.asarray(neural_obs).ravel() - self.mFR_curr) * self.sdFR_ratio
            # set the spike count of any unit that now has zero-mean with its original mean
            # This functionally removes it from the decoder. 
            neural_obs[self.zeromeanunits] = self.mFR[self.zeromeanunits] 

        # re-format as a column matrix
        neural_obs = np.mat(neural_obs.reshape(-1,1))

        # weighted_avg_lfc = kwargs.pop('weighted_avg_lfc', False)
        x = self.filt.state.mean
        
        # Run the filter
        self.filt(neural_obs, Bu=Bu, x_target=x_target, F=F)

        if assist_level > 0:
            # If a non-zero assist level is passed in, then it's interpreted as 
            # if weighted_avg_lfc:
            #     # calculates assist as:
            #     #   (1-assist)*(A*x + K*(y-C*A*x)) + assist*(A*x + B*u)
            #     # Note: the variable "Bu" here is actually equal to assist_level*B*u
            #     self.filt.state.mean = (1-assist_level)*self.filt.state.mean + assist_level*self.filt.A*x + Bu
            # else:
            #     self.filt.state.mean = (1-assist_level)*self.filt.state.mean + assist_level*Bu

            weighted_avg_lfc = int(weighted_avg_lfc)
            self.filt.state.mean = (1-assist_level)*self.filt.state.mean + weighted_avg_lfc*assist_level*self.filt.A*x + (1-weighted_avg_lfc)*assist_level*Bu

        # Bound cursor, if any hard bounds for states are applied
        if hasattr(self, 'bounder'):
            self.filt.state.mean = self.bounder(self.filt.state.mean, self.states)

        state = self.filt.get_mean()
        return state

    def decode(self, neural_obs, **kwargs):
        '''
        Decode multiple observations sequentially.

        Parameters
        ----------
        neural_obs: np.array of shape (# features, # observations)
            Independent neural observations are columns of the data
            matrix and are decoded sequentially
        kwargs: dict
            Container for special keyword-arguments for the specific decoding
            algorithm's 'predict'. 
        '''
        output = []
        n_obs = neural_obs.shape[1]
        for k in range(n_obs):
            self.predict(neural_obs[:,k], **kwargs)
            output.append(self.filt.get_mean())
        return np.vstack(output)

    def __str__(self):
        '''    Docstring    '''
        if hasattr(self, 'db_entry'):
            return self.db_entry.name
        else:
            return super(Decoder, self).__str__()

    @property
    def n_states(self):
        '''
        Return the number of states represented in the Decoder
        '''
        return len(self.states)

    @property
    def n_units(self):
        '''
        Return the number of units used in the decoder. Not sure what this 
        does for LFP decoders, i.e. decoders which extract multiple features from
        a single channel.
        '''
        return len(self.units)

    def __call__(self, obs_t, accumulate=True, **kwargs):
        '''
        This function does "rate-matching" to match the decoding rate to the 
        control rate of the plant. For instance, for cursor decoding using a 
        PPFDecoder, the decoder runs at 180Hz but the screen can only be updated
        at 60Hz, so the observations have to be presented 3 at a time. Similarly 
        a KFDecoder might run at 10 Hz, and the Decoder would have to accumulate
        observations over 6 iterations. 

        Parameters
        ----------
        obs_t: np.array of shape (# features, # subbins)
            Neural observation vector. If the decoding_rate of the Decoder is
            greater than the control rate of the plant (e.g. 60 Hz )
        kwargs: dictionary
            Algorithm-specific arguments to be given to the Decoder.predict method
        '''

        call_rate = self.call_rate
        decoding_rate = 1./self.binlen

        if decoding_rate >= call_rate:
            # Infer the number of sub-bins from the size of the spike counts mat to decode
            n_subbins = obs_t.shape[1]

            outputs = []
            for k in range(n_subbins):
                outputs.append(self.predict(obs_t[:,k], **kwargs))

            return np.vstack(outputs).T
        elif decoding_rate < call_rate:
            if accumulate:
                self.spike_counts += obs_t.reshape(-1, 1)
            else:
                self.spike_counts = obs_t.reshape(-1, 1)

            if self.bmicount == self.bminum - 1:
                # Update using spike counts
                self.bmicount = 0
                self.predict(self.spike_counts, **kwargs)
                # self.spike_counts = np.zeros([len(self.units), 1])
                self.spike_counts = np.zeros([self.n_features, 1])
            else:
                self.bmicount += 1
            return self.filt.get_mean().reshape(-1,1)

    def save(self, filename=''):
        '''
        Pickle the Decoder object to a file

        Parameters
        ----------
        filename: string, optional
            Filename to pickle the decoder to. If unspecified, a temporary file will be created.

        Returns
        -------
        filename: string
            filename of pickled Decoder object 
        '''
        if filename is not '':
            f = open(filename, 'w')
            pickle.dump(self, f)
            f.close()
            return filename
        else:
            import tempfile, cPickle
            tf2 = tempfile.NamedTemporaryFile(delete=False) 
            cPickle.dump(self, tf2)
            tf2.flush()
            return tf2.name

    def save_attrs(self, hdf_filename, table_name='task'):
        '''
        Save the attributes of the Decoder to the attributes of the specified HDF table

        Parameters
        ----------
        hdf_filename: string
            HDF filename to write data to
        table_name: string, default='task'
            Specify the table within the HDF file to set attributes in. 
        '''
        h5file = tables.openFile(hdf_filename, mode='a')
        table = getattr(h5file.root, table_name)
        for attr in self.filt.model_attrs:
            table.attrs[attr] = np.array(getattr(self.filt, attr))
        h5file.close()        

    @property
    def state_shape_rt(self):
        '''
        Create attribute to access the shape of the accumulating spike counts feature.
        '''
        return (self.n_states, self.n_subbins)


class BMISystem(object):
    '''
    This class encapsulates all of the BMI decoding computations, including assist and CLDA
    '''
    def __init__(self, decoder, learner, updater):
        '''
        Instantiate the BMISystem
        
        Parameters
        ----------
        decoder: bmi.Decoder instance
            The decoder maps spike counts into the "state" of the prosthesis
        learner: clda.Learner instance
            The learner estimates the "intended" prosthesis state from task goals.
        updater: clda.Updater instance
            The updater remaps the decoder parameters to better match sets of 
            observed spike counts and intended kinematics (from the learner)
        '''
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
        '''    Docstring    '''
        self.spike_counts = np.zeros([self.decoder.n_features, 1])

    def __call__(self, neural_obs, target_state, task_state, *args, **kwargs):
        '''
        Main function for all BMI functions, including running the decoder, adapting the decoder 
        and incorporating assist

        Parameters
        ----------
        neural_obs : np.ndarray, 
            The shape of neural_obs should be [n_units, n_obs]. If multiple observations are given, then
            the decoder will run multiple times before returning. 
        target_state : np.ndarray
            The assumed state that the subject is trying to drive the BMI toward, e.g. based on the 
            objective of the task 
        task_state : string
            State of the task. Used by CLDA so that assist is only applied during certain states,
            e.g. in some tasks, the target will be ambiguous during penalty states so CLDA should 
            ignore data during those epochs. 
        *args : tuple
            addional unnamed arguments
            This is mostly so that it won't complain if you make a mistake in calling the function
        **kwargs : dict
            Instance-specific arguments, e.g. RML/SmoothBatch require a 'half_life' parameter 
            that is not required of other CLDA methods. 

        Returns
        -------
        decoded_states : np.ndarray
            Columns of the array are vectors representing the decoder output as each of the 
            observations are decoded.
        update_flag : boolean
            Boolean to indicate whether the parameters of the Decoder have changed based on the
            current function call 
        '''
        n_units, n_obs = neural_obs.shape

        # If the target is specified as a 1D position, tile to match 
        # the number of dimensions as the neural features
        if np.ndim(target_state) == 1:
            target_state = np.tile(target_state, [n_obs, 1]).T

        decoded_states = np.zeros([self.decoder.n_states, n_obs])
        update_flag = False
        learn_flag = kwargs.pop('learn_flag', False)

        feature_type = kwargs.pop('feature_type')

        if feature_type in ['lfp_power', 'emg_amplitude']:
            accumulate = False
        else:
            accumulate = True

        for k in range(n_obs):
            neural_obs_k = neural_obs[:,k].reshape(-1,1)
            target_state_k = target_state[:,k]

            # NOTE: the conditional below is *only* for compatibility with older Carmena
            # lab data collected using a different MATLAB-based system. In all python cases, 
            # the task_state should never contain NaN values. 
            if np.any(np.isnan(target_state_k)): task_state = 'no_target' 

            # run the decoder
            prev_state = self.decoder.get_state()
            self.decoder(neural_obs_k, accumulate=accumulate, **kwargs)
            decoded_states[:,k] = self.decoder.get_state()

            # Determine whether the current state or previous state should be given to the learner
            if self.learner.input_state_index == 0:
                learner_state = decoded_states[:,k]
            elif self.learner.input_state_index == -1:
                learner_state = prev_state
            else:
                print "Not implemented yet: %d" % self.learner.input_state_index
                learner_state = prev_state

            # self.spike_counts += spike_obs_k
            if feature_type in ['lfp_power', 'emg_amplitude']:
                # hack to make to make lfp decoding work
                self.spike_counts = neural_obs_k
            else:
                self.spike_counts += neural_obs_k

            if learn_flag and self.decoder.bmicount == 0:
                self.learner(self.spike_counts.copy(), learner_state, target_state_k, 
                             decoded_states[:,k], task_state, state_order=self.decoder.ssm.state_order)
                self.reset_spike_counts()
            elif self.decoder.bmicount == 0:
                self.reset_spike_counts()
        
            new_params = None # by default, no new parameters are available

            if self.learner.is_ready():
                batch_data = self.learner.get_batch()
                batch_data['decoder'] = self.decoder
                # self.intended_kin, self.spike_counts_batch = self.learner.get_batch()
                # args = (self.intended_kin, self.spike_counts_batch, self.decoder)
                batch_size = batch_data['intended_kin'].shape[1]
                # batch_data['batch_time'] = batch_size * self.decoder.binlen
                kwargs.update(batch_data)

                if self.mp_updater:
                    args = ()
                    self.clda_input_queue.put(args, kwargs)
                    # Disable learner until parameter update is received
                    self.learner.disable() 
                else:
                    new_params = self.updater.calc(**kwargs)
                    if batch_size > 1: 
                        print "updating BMI"

            # If the updater is running in a separate process, check if a new 
            # parameter update is available
            if self.mp_updater:
                try:
                    new_params = self.clda_output_queue.get_nowait()
                except Queue.Empty:
                    pass
                except:
                    f = open(os.path.expandvars('$HOME/code/bmi3d/log/clda_log'), 'w')
                    traceback.print_exc(file=f)
                    f.close()

            # Update the decoder if new parameters are available
            if new_params is not None:
                self.decoder.update_params(new_params, **self.updater.update_kwargs)
                new_params['intended_kin'] = batch_data['intended_kin']
                new_params['spike_counts_batch'] = batch_data['spike_counts']

                self.learner.enable()
                update_flag = True

            # Update parameter history
            self.param_hist.append(new_params)

        # decoded_states = np.vstack(decoded_states).T
        return decoded_states, update_flag

    def __del__(self):
        '''
        Destructor for BMISystem. Stops any spawned processes, if any.
        '''
        # Stop updater if it's running in a separate process
        if self.mp_updater: 
            self.updater.stop()


class BMI(object):
    '''
    Legacy class, used only for unpickling super old Decoder objects. Ignore completely.
    '''
    pass