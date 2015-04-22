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
import datetime


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
        else:
            raise Exception(str(type(mean)))

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
        if isinstance(other, (int, float)) and (other == 0):
            return GaussianState(self.mean, self.cov)
        elif isinstance(other, (int, float)):
            return GaussianState(self.mean + other, self.cov)
        elif isinstance(other, GaussianState):
            return GaussianState(self.mean+other.mean, self.cov+other.cov)
        elif isinstance(other, np.matrix) and other.shape == self.mean.shape:
            return GaussianState(self.mean + other, self.cov)
        else:
            # print other
            raise ValueError("Gaussian state: cannot add type :%s" % type(other))


class GaussianStateHMM(object):
    '''
    General hidden Markov model decoder where the state is represented as a Gaussian random vector
    '''
    model_attrs = []

    # List out the attributes to save at pickle time. Might not want this to be every attribute of the decoder (e.g., no point in saving the state of the BMI at pickle-time)
    attrs_to_pickle = []
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
        '''
        Prior prediction of the hidden states using for linear directed random walk model
        x_{t+1} = Ax_t + c_t + w_t
            x_t = previous state
            c_t = control input (the "directed" part of the model)
            w_t = process noise (the "random walk" part of the model)

        Parameters
        ----------
        state : GaussianState instance
            State estimate and estimator covariance of current state
        u : np.mat, optional, default=None
            An assistive control input. Requires the filter to have an input matrix attribute, B
        Bu : np.mat of shape (N, 1)
            Assistive control input which is precomputed to already account for the control input matrix
        target_state : np.mat of shape (N, 1)
            Optimal value for x_t (defined by external factors, i.e. the task being performed)
        F : np.mat of shape (B.shape[1], N)
            Feedback control gains. Used to compute u_t = BF(x^* - x_t)
        

        Returns
        -------
        GaussianState instance
            Represents the mean and estimator covariance of the new state estimate
        '''
        A = self.A

        if not (Bu is None):
            c_t = Bu
        elif not (u is None):
            c_t = self.B * u
        elif not (target_state is None):
            B = self.B
            if F is None:
                F = self.F
            # if not np.all(target_state[:-1,:] == 0):
            #     import pdb; pdb.set_trace()
            A = A - B*F
            c_t = B*F*target_state
        else:
            c_t = 0

        return A * state + c_t + self.state_noise

    def __eq__(self, other):
        '''
        Determine equality of two GaussianStateHMM instances
        '''
        # import train
        return GaussianStateHMM.obj_eq(self, other, self.model_attrs)

    def __sub__(self, other):
        '''
        Subtract the model attributes of two GaussianStateHMM instances. Used to determine approximate equality, i.e., equality modulo floating point error
        '''
        # import train
        return GaussianStateHMM.obj_diff(self, other, self.model_attrs)

    @staticmethod
    def obj_eq(self, other, attrs=[]):
        '''
        Determine if two objects have mattching array attributes

        Parameters
        ----------
        other : object
            If objects are not the same type, False is returned
        attrs : list, optional
            List of attributes to compare for equality. Only attributes that are common to both objects are used.
            The attributes should be np.array or similar as np.array_equal is used to determine equality

        Returns
        -------
        bool 
            True value returned indicates equality between objects for the specified attributes
        '''
        if isinstance(other, type(self)):
            attrs_eq = filter(lambda y: y in other.__dict__, filter(lambda x: x in self.__dict__, attrs))
            equal = map(lambda attr: np.array_equal(getattr(self, attr), getattr(other, attr)), attrs_eq)
            return np.all(equal)
        else:
            return False
    
    @staticmethod
    def obj_diff(self, other, attrs=[]):
        '''
        Calculate the difference of the two objects w.r.t the specified attributes

        Parameters
        ----------
        other : object
            If objects are not the same type, False is returned
        attrs : list, optional
            List of attributes to compare for equality. Only attributes that are common to both objects are used.
            The attributes should be np.array or similar as np.array_equal is used to determine equality

        Returns
        -------
        np.array
            The difference between each of the specified 'attrs'
        '''
        if isinstance(other, type(self)):
            attrs_eq = filter(lambda y: y in other.__dict__, filter(lambda x: x in self.__dict__, attrs))
            diff = map(lambda attr: getattr(self, attr) - getattr(other, attr), attrs_eq)
            return np.array(diff)
        else:
            return False

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

    def __getstate__(self):
        data_to_pickle = dict()
        for attr in self.attrs_to_pickle:
            try:
                data_to_pickle[attr] = getattr(self, attr)
            except:
                print "GaussianStateHMM: could not pickle attribute %s" % attr
        return data_to_pickle


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
            import state_space_models
            self.ssm = state_space_models.StateSpaceEndptVel2D()
            # self.ssm = train.endpt_2D_state_space

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
        elif isinstance(idx, str) or isinstance(idx, unicode):
            idx = self.states.index(idx)
            self.filt.state.mean[idx, 0] = value
        elif np.iterable(idx):
            [self.__setitem__(k, val) for k, val in izip(idx, value)]
        else:
            raise ValueError("Decoder: Improper index type: %" % type(idx))

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

        if not hasattr(self, 'n_features'):
            self.n_features = len(self.units)

        self._pickle_init()

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

    def predict(self, neural_obs, assist_level=0.0, weighted_avg_lfc=False, **kwargs):
        """
        Decode the spikes

        Parameters
        ----------
        neural_obs: np.array of shape (N,) or (N, 1)
            One time-point worth of neural features to decode
        assist_level: float
            Weight given to the assist term. This variable name may be a slight misnomer, a more appropriate term might be 'reweight_factor'
        Bu: np.mat of shape (N, 1)
            Assist vector to be added on to the Decoder state. Must be of the same dimension 
            as the state vector.
        kwargs: dict
            Mostly for kwargs function call compatibility
        """
        if assist_level > 0 and 'x_assist' not in kwargs:
            raise ValueError("Assist cannot be used if the forcing term is not specified!")

        # re-normalize the variance of the spike observations, if nec
        if hasattr(self, 'zscore') and self.zscore:
            neural_obs = (np.asarray(neural_obs).ravel() - self.mFR_curr) * self.sdFR_ratio
            # set the spike count of any unit that now has zero-mean with its original mean
            # This functionally removes it from the decoder. 
            neural_obs[self.zeromeanunits] = self.mFR[self.zeromeanunits] 

        # re-format as a column matrix
        neural_obs = np.mat(neural_obs.reshape(-1,1))

        x = self.filt.state.mean
        
        # Run the filter
        self.filt(neural_obs, **kwargs)

        if assist_level > 0:
            weighted_avg_lfc = int(weighted_avg_lfc)
            x_assist = kwargs.pop('x_assist')
            self.filt.state.mean = (1-assist_level)*self.filt.state.mean + assist_level * x_assist

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

    def __call__(self, obs_t, **kwargs):
        '''
        Wrapper for the 'predict' method

        Parameters
        ----------
        obs_t: np.array of shape (# features, # subbins)
            Neural observation vector. If the decoding_rate of the Decoder is
            greater than the control rate of the plant (e.g. 60 Hz )
        kwargs: dictionary
            Algorithm-specific arguments to be given to the Decoder.predict method
        '''

        self.predict(obs_t, **kwargs)
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
    def __init__(self, decoder, learner, updater, feature_accumulator):
        '''
        Instantiate the BMISystem
        
        Parameters
        ----------
        decoder : bmi.Decoder instance
            The decoder maps spike counts into the "state" of the prosthesis
        learner : clda.Learner instance
            The learner estimates the "intended" prosthesis state from task goals.
        updater : clda.Updater instance
            The updater remaps the decoder parameters to better match sets of 
            observed spike counts and intended kinematics (from the learner)
        feature_accumulator : accumulator.FeatureAccumulator instance
            Combines features across time if necesary to perform rate matching 
            between the task rate and the decoder rate.

        Returns
        -------
        BMISystem instance
        '''
        self.decoder = decoder 
        self.learner = learner
        self.updater = updater
        self.feature_accumulator = feature_accumulator
        self.param_hist = []

        if self.updater is not None:
            self.updater.init(self.decoder)

    def __call__(self, neural_obs, target_state, task_state, learn_flag=False, **kwargs):
        '''
        Main function for all BMI functions, including running the decoder, adapting the decoder 
        and incorporating assistive control inputs

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
        learn_flag : bool, optional, default=True
            Boolean specifying whether the decoder should update based on intention estimates
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
        if np.ndim(target_state) == 1 or (target_state.shape[1] == 1 and n_obs > 1):
            target_state = np.tile(target_state, [1, n_obs])


        decoded_states = np.zeros([self.decoder.n_states, n_obs])
        update_flag = False

        for k in range(n_obs):
            neural_obs_k = neural_obs[:,k].reshape(-1,1)
            target_state_k = target_state[:,k]

            # NOTE: the conditional below is *only* for compatibility with older Carmena
            # lab data collected using a different MATLAB-based system. In all python cases, 
            # the task_state should never contain NaN values. 
            if np.any(np.isnan(target_state_k)): task_state = 'no_target' 

            #################################
            ## Decode the current observation
            #################################
            decodable_obs, decode = self.feature_accumulator(neural_obs_k)

            if decode: # if a new decodable observation is available from the feature accumulator
                prev_state = self.decoder.get_state()
                
                self.decoder(decodable_obs, **kwargs)

                # Determine whether the current state or previous state should be given to the learner
                if self.learner.input_state_index == 0:
                    learner_state = self.decoder.get_state()
                elif self.learner.input_state_index == -1:
                    learner_state = prev_state
                else:
                    print "Not implemented yet: %d" % self.learner.input_state_index
                    learner_state = prev_state

                if learn_flag:
                    self.learner(decodable_obs.copy(), learner_state, target_state_k, self.decoder.get_state(), task_state, state_order=self.decoder.ssm.state_order)

            decoded_states[:,k] = self.decoder.get_state()        

            ############################
            ## Update decoder parameters
            ############################
            if self.learner.is_ready():
                batch_data = self.learner.get_batch()
                batch_data['decoder'] = self.decoder
                kwargs.update(batch_data)

                self.updater(**kwargs)
                self.learner.disable() 

            new_params = None # by default, no new parameters are available
            if not (self.updater is None):
                new_params = self.updater.get_result()

            # Update the decoder if new parameters are available
            if not (new_params is None):
                self.decoder.update_params(new_params, **self.updater.update_kwargs)
                new_params['intended_kin'] = batch_data['intended_kin']
                new_params['spike_counts_batch'] = batch_data['spike_counts']

                self.learner.enable()
                update_flag = True

            # Update parameter history
            self.param_hist.append(new_params)

        return decoded_states, update_flag


class BMILoop(object):
    '''
    Container class/interface definition for BMI tasks. Intended to be used with multiple inheritance structure paired with riglib.experiment classes
    '''
    static_states = [] # states in which the decoder is not run
    decoder_sequence = ''

    def init(self):
        '''
        Secondary init function. Finishes initializing the task after all the 
        constructors have run and all the requried attributes have been declared
        for the task to operate. 
        '''
        # Initialize the decoder
        self.load_decoder()
        self.init_decoder_state()

        # Declare data attributes to be stored in the sinks every iteration of the FSM
        self.add_dtype('loop_time', 'f8', (1,))
        self.add_dtype('decoder_state', 'f8', (self.decoder.n_states, 1))
        self.add_dtype('internal_decoder_state', 'f8', self.decoder.state_shape_rt)
        self.add_dtype('target_state', 'f8', self.decoder.state_shape_rt)
        self.add_dtype('update_bmi', 'f8', (1,))

        # Construct the sub-pieces of the BMI system
        self.create_assister()
        self.create_feature_extractor()
        self.create_feature_accumulator()        
        self.create_goal_calculator()
        self.create_learner()
        self.create_updater()
        self.create_bmi_system()

        super(BMILoop, self).init()        

    def create_bmi_system(self):
        self.bmi_system = BMISystem(self.decoder, self.learner, self.updater, self.feature_accumulator)

    def load_decoder(self):
        '''
        Shell function. In tasks launched from the GUI with the BMI feature 
        enabled, the decoder attribute is automatically added to the task. This
        is for simulation purposes only (or if you want to make a version that
        launches from the command line)
        '''
        pass

    def init_decoder_state(self):
        '''
        Initialize the state of the decoder to match the initial state of the plant
        '''
        self.decoder.filt._init_state()
        try:
            self.decoder['q'] = self.plant.get_intrinsic_coordinates()
        except:
            print self.plant.get_intrinsic_coordinates()
            print self.decoder['q']
            raise Exception("Error initializing decoder state")
        self.init_decoder_mean = self.decoder.filt.state.mean

        self.decoder.set_call_rate(1./self.update_rate)

    def create_assister(self):
        '''
        The 'assister' is a callable object which, for the specific plant being controlled,
        will drive the plant toward the specified target state of the task. 
        '''
        self.assister = None

    def create_feature_accumulator(self):
        '''
        Instantiate the feature accumulator used to implement rate matching between the Decoder and the task,
        e.g. using a 10 Hz KFDecoder in a 60 Hz task
        '''
        import accumulator
        feature_shape = [self.decoder.n_features, 1]
        feature_dtype = np.float64
        acc_len = int(self.decoder.binlen / self.update_rate)
        acc_len = max(1, acc_len)

        if self.extractor.feature_type in ['lfp_power', 'emg_amplitude']:
            self.feature_accumulator = accumulator.NullAccumulator(acc_len)
        else:
            self.feature_accumulator = accumulator.RectWindowSpikeRateEstimator(acc_len, feature_shape, feature_dtype)

    def create_goal_calculator(self):
        '''
        The 'goal_calculator' is a callable object which will define the optimal state for the Decoder 
        to be in for this particular task. This object is necessary for CLDA (to estimate the "error" of the decoder
        in order to adapt it) and for any assistive control (for the 'machine' controller to determine where to 
        drive the plant 
        '''
        self.goal_calculator = None

    def create_feature_extractor(self):
        '''
        Create the feature extractor object. The feature extractor takes raw neural data from the streaming processor
        (e.g., spike timestamps) and outputs a decodable observation vector (e.g., counts of spikes in last 100ms from each unit)
        '''
        import extractor
        if hasattr(self.decoder, 'extractor_cls') and hasattr(self.decoder, 'extractor_kwargs'):
            self.extractor = self.decoder.extractor_cls(self.neurondata, **self.decoder.extractor_kwargs)
        else:
            # if using an older decoder that doesn't have extractor_cls and 
            # extractor_kwargs as attributes, then create a BinnedSpikeCountsExtractor by default
            self.extractor = extractor.BinnedSpikeCountsExtractor(self.neurondata, 
                n_subbins=self.decoder.n_subbins, units=self.decoder.units)

        self._add_feature_extractor_dtype()

    def _add_feature_extractor_dtype(self):
        '''
        Helper function to add the datatype of the extractor output to be saved in the HDF file. Uses a separate function 
        so that simulations can overwrite.
        '''
        if isinstance(self.extractor.feature_dtype, tuple): # Feature extractor only returns 1 type
            self.add_dtype(*self.extractor.feature_dtype)
        else:
            for x in self.extractor.feature_dtype: # Feature extractor returns multiple named fields
                self.add_dtype(*x)

    def create_learner(self):
        '''
        The "learner" uses knowledge of the task goals to determine the "intended" 
        action of the BMI subject and pairs this intention estimation with actual observations.
        '''
        import clda
        self.learn_flag = False
        self.learner = clda.DumbLearner()

    def create_updater(self):
        '''
        The "updater" uses the output batches of data from the learner and an update rule to 
        alter the decoder parameters to better match the intention estimates.
        '''
        self.updater = None

    def call_decoder(self, neural_obs, target_state, **kwargs):
        '''
        Run the decoder computations

        Parameters
        ----------
        neural_obs : np.array of shape (n_features, n_subbins)
            n_features is the number of neural features the decoder is expecting to decode from.
            n_subbins is the number of simultaneous observations which will be decoded (typically 1)
        target_state : np.array of shape (n_states, 1)
            The current optimal state to be in to accomplish the task. In this function call, this gets
            used when adapting the decoder using CLDA
        kwargs : optional keyword arguments
            Optional arguments to CLDA, assist, etc.
        '''
        # Get the decoder output
        decoder_output, update_flag = self.bmi_system(neural_obs, target_state, self.state, learn_flag=self.learn_flag, **kwargs)
        self.task_data['update_bmi'] = int(update_flag)
        return decoder_output

    def get_features(self):
        '''
        Run the feature extractor to get any new features to be decoded. Called by move_plant
        '''
        start_time = self.get_time()        
        return self.extractor(start_time)

    def move_plant(self, **kwargs):
        '''
        The main functions to retrieve raw observations from the neural data source and convert them to movement of the plant

        Parameters
        ----------
        **kwargs : optional keyword arguments
            optional arguments for the decoder, assist, CLDA, etc. fed to the BMISystem

        Returns
        -------
        decoder_state : np.mat
            (N, 1) vector representing the state decoded by the BMI
        '''
        # Run the feature extractor
        feature_data = self.get_features()

        # Save the "neural features" (e.g., spike counts vector) to HDF file
        for key, val in feature_data.items():
            self.task_data[key] = val

        # Determine the target_state and save to file
        if self.current_assist_level > 0 or self.learn_flag:
            target_state = self.get_target_BMI_state(self.decoder.states)
        else:
            target_state = np.ones([self.decoder.n_states, self.decoder.n_subbins]) * np.nan


        # Determine the assistive control inputs to the Decoder
        if self.current_assist_level > 0:
            current_state = self.get_current_state()
            if target_state.shape[1] > 1:
                assist_kwargs = self.assister(current_state, target_state[:,0].reshape(-1,1), self.current_assist_level, mode=self.state)
            else:
                assist_kwargs = self.assister(current_state, target_state, self.current_assist_level, mode=self.state)
            kwargs.update(assist_kwargs)

        # Run the decoder
        if self.state not in self.static_states:
            neural_features = feature_data[self.extractor.feature_type]
            self.task_data['internal_decoder_state'] = self.call_decoder(neural_features, target_state, **kwargs)

        # Drive the plant to the decoded state, if permitted by the constraints of the plant
        # If not possible, plant.drive should also take care of setting the decoder's 
        # state as close as possible to physical reality
        self.plant.drive(self.decoder)

        self.task_data['decoder_state'] = decoder_state = self.decoder.get_state(shape=(-1,1))
        return decoder_state

    def get_current_state(self):
        '''
        In most cases, the current state of the plant needed for calculating assistive control inputs will be stored in the decoder
        '''
        return self.decoder.filt.state.mean

    def get_target_BMI_state(self, *args):
        '''
        Run the goal calculator to determine what the target state of the task is.
        Since this is not a real task, this function must be 
        overridden in child classes if any of the assist/CLDA functionality is to be used.
        '''
        raise NotImplementedError

    def _cycle(self):
        # save loop time to HDF file
        self.task_data['loop_time'] = self.iter_time()
        super(BMILoop, self)._cycle()

    def enable_clda(self):
        print "CLDA enabled"
        self.learn_flag = True

    def disable_clda(self):
        print "CLDA disabled after %d successful trials" % self.calc_state_occurrences('reward')
        self.learn_flag = False

    def cleanup_hdf(self):
        '''
        Re-open the HDF file and save any extra task data kept in RAM
        '''
        super(CLDAMouseSpeller, self).cleanup_hdf()
        log_file = open(os.path.join(os.getenv("HOME"), 'code/bmi3d/log/clda_log'), 'w')
        log_file.write(str(self.state) + '\n')
        try:
            if len(self.bmi_system.param_hist) > 0:
                ignore_none = self.learner.batch_size > 1
                log_file.write('Ignoring "None" values: %s\n' % str(ignore_none))
                clda.write_clda_data_to_hdf_table(
                    self.h5file.name, self.bmi_system.param_hist, 
                    ignore_none=ignore_none)
        except:
            import traceback
            traceback.print_exc(file=log_file)
        log_file.close()

    def cleanup(self, database, saveid, **kwargs):
        super(CLDAMouseSpeller, self).cleanup(database, saveid, **kwargs)

        # Open a log file in case of error b/c errors not visible to console
        # at this point
        f = open(os.path.join(os.getenv('HOME'), 'code/bmi3d/log/clda_cleanup_log'), 'w')
        f.write('Opening log file\n')
        
        # save out the parameter history and new decoder unless task was stopped
        # before 1st update
        try:
            f.write('# of paramter updates: %d\n' % len(self.bmi_system.param_hist))

            if len(self.bmi_system.param_hist) > 0:
                # create name for new decoder 
                now = datetime.datetime.now()
                decoder_name = self.decoder_sequence + now.strftime('%m%d%H%M')

                # Pickle the decoder
                decoder_tempfilename = self.decoder.save()

                # Link the pickled decoder file to the associated task entry in the database
                dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
                if dbname == 'default':
                    database.save_bmi(decoder_name, saveid, decoder_tempfilename)
                else:
                    database.save_bmi(decoder_name, saveid, decoder_tempfilename, dbname=dbname)
        except:
            traceback.print_exc(file=f)
        f.close()


class BMI(object):
    '''
    Legacy class, used only for unpickling super old Decoder objects. Ignore completely.
    '''
    pass