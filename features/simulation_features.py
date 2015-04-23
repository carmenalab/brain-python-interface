'''
Features for use in simulation tasks
'''

import time
import tempfile
import random
import traceback
import numpy as np
import fnmatch
import os
import subprocess
from riglib.experiment import traits

from riglib.bmi import accumulator, assist, bmi, clda, extractor, feedback_controllers, goal_calculators, robot_arms, sim_neurons, kfdecoder, ppfdecoder, state_space_models, train
from riglib.bmi.sim_neurons import KalmanEncoder

import pickle


class FakeHDF(object):
    def __init__(self):
        self.msgs = []

    def sendMsg(self, msg):
        self.msgs.append(msg)


class SimHDF(object):
    '''
    An interface-compatbile HDF for simulations which do not require saving an
    HDF file
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for SimHDF feature

        Parameters
        ----------
        args, kwargs: None necessary

        Returns
        -------
        SimHDF instance
        '''
        from collections import defaultdict
        self.data = defaultdict(list)
        self.task_data_hist = []
        self.msgs = []        
        self.hdf = FakeHDF()

        super(SimHDF, self).__init__(*args, **kwargs)

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' creates a fake task data variable so that 
        code expecting SaveHDF runs smoothly.
        '''
        super(SimHDF, self).init()
        self.dtype = np.dtype(self.dtype)
        self.task_data = np.zeros((1,), dtype=self.dtype)

    def sendMsg(self, msg):
        '''
        Simulate the "message" table of the HDF file associated with each source

        Parameters
        ----------
        msg: string
            Message to store

        Returns
        -------
        None
        '''
        self.msgs.append((msg, -1))

    def _cycle(self):
        super(SimHDF, self)._cycle()
        self.task_data_hist.append(self.task_data.copy())


class SimTime(object):
    '''
    An accelerator so that simulations can run faster than real time (the task doesn't try to 'sleep' between loop iterations)
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for SimTime

        Parameters
        ----------
        *args, **kwargs: passed to parent (super) constructor

        Returns
        -------
        None
        '''
        super(SimTime, self).__init__(*args, **kwargs)
        self.start_time = 0.

    def get_time(self):
        '''
        Simulates time based on Delta*cycle_count, where the update_rate is specified as an instance attribute
        '''
        try:
            return self.cycle_count * self.update_rate
        except:
            # loop_counter has not been initialized yet, return 0
            return 0

    @property 
    def update_rate(self):
        '''
        Attribute for update rate of task. Using @property in case any future modifications
        decide to change fps on initialization
        '''
        return 1./60


class SimNeuralEnc(object):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'fb_ctrl'):
            self.fb_ctrl = kwargs.pop('fb_ctrl')
        if not hasattr(self, 'ssm'):
            self.ssm = kwargs.pop('ssm')
        super(SimNeuralEnc, self).__init__(*args, **kwargs)

    def init(self):
        self._init_neural_encoder()
        self.wait_time = 0
        self.pause = False
        super(SimNeuralEnc, self).init()

    def _init_neural_encoder(self):
        raise NotImplementedError


class SimKalmanEnc(SimNeuralEnc):
    def _init_neural_encoder(self):
        ## Simulation neural encoder
        n_features = 20
        self.encoder = KalmanEncoder(self.ssm, n_features)

    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.SimDirectObsExtractor(self.fb_ctrl, self.encoder, 
            n_subbins=self.decoder.n_subbins, units=self.decoder.units, task=self)
        self._add_feature_extractor_dtype()

    def create_feature_accumulator(self):
        '''
        Instantiate the feature accumulator used to implement rate matching between the Decoder and the task,
        e.g. using a 10 Hz KFDecoder in a 60 Hz task
        '''
        from riglib.bmi import accumulator
        feature_shape = [self.decoder.n_features, 1]
        feature_dtype = np.float64
        acc_len = int(self.decoder.binlen / self.update_rate)
        acc_len = max(1, acc_len)

        self.feature_accumulator = accumulator.NullAccumulator(acc_len)


class SimCosineTunedEnc(SimNeuralEnc):
    def _init_neural_encoder(self):
        ## Simulation neural encoder
        from riglib.bmi.sim_neurons import CLDASimCosEnc
        self.encoder = CLDASimCosEnc(return_ts=True)    
        
    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.SimBinnedSpikeCountsExtractor(self.fb_ctrl, self.encoder, 
            n_subbins=self.decoder.n_subbins, units=self.decoder.units, task=self)
        self._add_feature_extractor_dtype()


class SimCosineTunedPointProc(SimNeuralEnc):
    def _init_neural_encoder(self):
        # from riglib.bmi import sim_neurons
        # decoder_fname = '/storage/decoders/grom20150102_14_BPPF01021655.pkl'
        # decoder = pickle.load(open(decoder_fname))

        # beta = np.array(decoder.filt.C)
        # dt = decoder.filt.dt 
        # self.encoder = sim_neurons.CLDASimPointProcessEnsemble(beta, dt)


        # 2) create a fake beta
        const = -1.6
        alpha = 0.10
        n_cells = 30
        angles = np.linspace(0, 2*np.pi, n_cells)
        beta = np.zeros([n_cells, 7])
        beta[:,-1] = -1.6
        beta[:,3] = np.cos(angles)*alpha
        beta[:,5] = np.sin(angles)*alpha
        beta = np.vstack([np.cos(angles)*alpha, np.sin(angles)*alpha, np.ones_like(angles)*const]).T

        beta_full = np.zeros([n_cells, 7])
        beta_full[:,[3,5,6]] = beta
        # dec = train._train_PPFDecoder_sim_known_beta(beta_full, encoder.units, dt=dt)

        # create the encoder
        dt = 1./180
        self.encoder = sim_neurons.CLDASimPointProcessEnsemble(beta_full, dt)
        self.beta_full = beta_full

    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.SimBinnedSpikeCountsExtractor(self.fb_ctrl, self.encoder, 
            n_subbins=self.decoder.n_subbins, units=self.decoder.units, task=self)
        self._add_feature_extractor_dtype()        

#############################
##### Simulation Decoders
#############################

class SimKFDecoderSup(object):
    '''
    Construct a KFDecoder based on encoder output in response to states simulated according to the state space model's process noise
    '''
    def load_decoder(self):
        '''
        Instantiate the neural encoder and "train" the decoder
        '''
        print "Creating simulation decoder.."
        encoder = self.encoder
        n_samples = 20000
        units = self.encoder.get_units()
        n_units = len(units)

        # draw samples from the W distribution
        ssm = self.ssm
        A, _, W = ssm.get_ssm_matrices()
        mean = np.zeros(A.shape[0])
        mean[-1] = 1
        state_samples = np.random.multivariate_normal(mean, 100*W, n_samples)

        spike_counts = np.zeros([n_units, n_samples])
        self.encoder.call_ds_rate = 1
        for k in range(n_samples):
            spike_counts[:,k] = np.array(self.encoder(state_samples[k])).ravel()

        kin = state_samples.T

        self.decoder = train.train_KFDecoder_abstract(ssm, kin, spike_counts, units, 0.1)
        self.encoder.call_ds_rate = 6
        super(SimKFDecoderSup, self).load_decoder()

class SimKFDecoderShuffled(object):
    '''
    Construct a KFDecoder based on encoder output in response to states simulated according to the state space model's process noise
    '''
    def load_decoder(self):
        '''
        Instantiate the neural encoder and "train" the decoder
        '''
        print "Creating simulation decoder.."
        encoder = self.encoder
        n_samples = 20000
        units = self.encoder.get_units()
        n_units = len(units)

        # draw samples from the W distribution
        ssm = self.ssm
        A, _, W = ssm.get_ssm_matrices()
        mean = np.zeros(A.shape[0])
        mean[-1] = 1
        state_samples = np.random.multivariate_normal(mean, 100*W, n_samples)

        spike_counts = np.zeros([n_units, n_samples])
        self.encoder.call_ds_rate = 1
        for k in range(n_samples):
            spike_counts[:,k] = np.array(self.encoder(state_samples[k])).ravel()

        inds = np.arange(spike_counts.shape[0])
        np.random.shuffle(inds)
        spike_counts = spike_counts[inds,:]

        kin = state_samples.T

        self.decoder = train.train_KFDecoder_abstract(ssm, kin, spike_counts, units, 0.1)
        self.encoder.call_ds_rate = 6
        super(SimKFDecoderShuffled, self).load_decoder()

class SimKFDecoderRandom(object):
    def load_decoder(self):
        '''
        Create a 'seed' decoder for the simulation which is simply randomly initialized
        '''
        from riglib.bmi import state_space_models
        ssm = state_space_models.StateSpaceEndptVel2D()
        self.decoder = train.rand_KFDecoder(ssm, self.encoder.get_units())
        super(SimKFDecoderRandom, self).load_decoder()


class SimPPFDecoderCursor(object):
    def load_decoder(self):
        # decoder_fname = '/storage/decoders/grom20150102_14_BPPF01021655.pkl'
        # decoder = pickle.load(open(decoder_fname))

        # beta = decoder.filt.C
        # self.init_beta = beta.copy()

        # inds = np.arange(beta.shape[0])
        # # np.random.shuffle(inds)
        # beta_dec = beta[inds, :]

        # dt = decoder.filt.dt 
        # self._init_neural_encoder()

        self.decoder = train._train_PPFDecoder_sim_known_beta(self.beta_full, self.encoder.units, dt=1./180)

class SimPPFDecoderCursorShuffled(object):
    def load_decoder(self):
        # decoder_fname = '/storage/decoders/grom20150102_14_BPPF01021655.pkl'
        # decoder = pickle.load(open(decoder_fname))

        # beta = decoder.filt.C
        # self.init_beta = beta.copy()

        inds = np.arange(self.beta_full.shape[0])
        np.random.shuffle(inds)
        self.shuffling_inds = inds
        # beta_dec = beta[inds, :]

        # dt = decoder.filt.dt 
        # self._init_neural_encoder()

        self.decoder = train._train_PPFDecoder_sim_known_beta(self.beta_full[inds], self.encoder.units, dt=1./180)





