'''
Features for use in simulation tasks
'''
import numpy as np
from riglib.bmi import clda, extractor, sim_neurons, kfdecoder, train
from riglib.bmi.sim_neurons import KalmanEncoder

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


class SimClock(object):
    def __init__(self, *args, **kwargs):
        super(SimClock, self).__init__(*args, **kwargs)

    def tick(self, *args, **kwargs):
        pass

class SimClockTick(object):
    '''
    Summary: A simulation pygame.clock to use in simulations that inherit from experiment.Experiment, to overwrite
    the pygame.clock.tick in the ._cycle function ( self.clock.tick(self.fps) )
    '''
    def __init__(self, *args, **kwargs):
        '''
        Summary: Constructor for SimClock
        Input param: *args:
        Input param: **kwargs:
        Output param: 
        '''
        super(SimClockTick, self).__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        self.clock = SimClock()
        super(SimClockTick, self).init(*args, **kwargs)


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
            if not (self.cycle_count % (60*10)):
                print(self.cycle_count/(60*10.))
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
            self.fb_ctrl = kwargs.pop('fb_ctrl', None)
        if not hasattr(self, 'ssm'):
            self.ssm = kwargs.pop('ssm', None)
        super(SimNeuralEnc, self).__init__(*args, **kwargs)

    def init(self,):
        self._init_neural_encoder()
        self.wait_time = 0
        self.pause = False
        print('neural encoder init function ', self)
        super(SimNeuralEnc, self).init()

    def change_enc_ssm(self, new_ssm):
        self.encoder_old = self.encoder
        C_old = self.encoder_old.C.copy()
        Q = self.encoder_old.Q.copy()
        old_ssm = np.array(self.encoder_old.ssm.state_names).copy()

        old_to_new_map = np.squeeze(np.array([np.nonzero(np.array(new_ssm.state_names)==st)[0] for st in old_ssm]))

        self.ssm = new_ssm
        self._init_neural_encoder()

        self.encoder.C[:,old_to_new_map] =  C_old
        self.encoder.Q = Q

    def _init_neural_encoder(self):
        raise NotImplementedError


class SimKalmanEnc(SimNeuralEnc):
    def _init_neural_encoder(self):
        n_features = 50
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
        self.feature_accumulator = accumulator.RectWindowSpikeRateEstimator(acc_len, feature_shape, feature_dtype)


class SimCosineTunedEnc(SimNeuralEnc):
    
    def _init_neural_encoder(self):
        ## Simulation neural encoder
        from riglib.bmi.sim_neurons import GenericCosEnc#CLDASimCosEnc
        print('SimCosineTunedEnc SSM:', self.ssm)
        self.encoder = GenericCosEnc(self.sim_C, self.ssm, return_ts=True, DT=0.1, call_ds_rate=6)
        
    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.SimBinnedSpikeCountsExtractor(self.fb_ctrl, self.encoder, 
            n_subbins=self.decoder.n_subbins, units=self.decoder.units, task=self)
        self._add_feature_extractor_dtype()

class SimNormCosineTunedEnc(SimNeuralEnc):

    def _init_neural_encoder(self):
        from riglib.bmi.sim_neurons import NormalizedCosEnc
        self.encoder = NormalizedCosEnc(self.plant.endpt_bounds, self.sim_C, self.ssm, spike=True, return_ts=False, 
            DT=self.update_rate, tick=self.update_rate, gain=self.fov)
    
    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.SimDirectObsExtractor(self.fb_ctrl, self.encoder, 
            n_subbins=self.decoder.n_subbins, units=self.decoder.units, task=self)
        self._add_feature_extractor_dtype()

class SimLFPCosineTunedEnc(SimNeuralEnc):

    bands = [(51, 100)]

    def _init_neural_encoder(self):
        from riglib.bmi.sim_neurons import NormalizedCosEnc
        self.encoder = NormalizedCosEnc(self.plant.endpt_bounds, self.sim_C, self.ssm, spike=False, return_ts=False, 
            DT=self.update_rate, tick=self.update_rate, n_bands=len(self.bands), gain=self.fov)
    
    def create_feature_extractor(self):
        '''
        Create the feature extractor object
        '''
        self.extractor = extractor.SimPowerExtractor(self.fb_ctrl, self.encoder, 
            channels=self.decoder.channels, bands=self.bands, task=self)
        self._add_feature_extractor_dtype()

class SimFAEnc(SimCosineTunedEnc):
    def __init__(self, *args, **kwargs):
        self.FACosEnc_kwargs = kwargs.pop('SimFAEnc_kwargs', dict())
        super(SimFAEnc, self).__init__(*args, **kwargs)

    def _init_neural_encoder(self):
        if 'encoder_fname' in self.FACosEnc_kwargs:
            import pickle
            self.encoder = pickle.load(open(self.FACosEnc_kwargs['encoder_fname']))
            print('using saved encoder;')
            
            if 'wt_sources' in self.FACosEnc_kwargs:
                self.encoder.wt_sources = self.FACosEnc_kwargs['wt_sources']
                print('setting new weights: ', self.encoder.wt_sources)
        
        else:
            from riglib.bmi.sim_neurons import FACosEnc
            self.encoder = FACosEnc(self.sim_C, self.ssm, return_ts=True, DT=0.1, call_ds_rate=6, **self.FACosEnc_kwargs)


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


class SimKFDecoder(object):
    '''
    General class for all SimDFDecoder classes
    '''
    def __init__(self, *args, **kwargs):
        super(SimKFDecoder, self).__init__(*args, **kwargs)

    def change_dec_ssm(self):
        decoder_old = self.decoder_old
        ssm_old = decoder_old.ssm
        
        #Map new stuff
        old_to_new_map = np.squeeze(np.array([np.nonzero(np.array(self.ssm.state_names)==st)[0] for st in ssm_old.state_names]))

        #Update C: 
        C = self.decoder.kf.C.copy()
        C[:,old_to_new_map] = decoder_old.filt.C

        #Keep current dtype format: 
        dtype_new = self.dtype[:]
       
        #New KF: 
        kf = kfdecoder.KalmanFilter(self.decoder.kf.A, self.decoder.kf.W, C, self.decoder.kf.Q, is_stochastic=self.ssm.is_stochastic)
        #New decoder: 
        self.decoder = kfdecoder.KFDecoder(kf, self.decoder.units, self.ssm, mFR=self.decoder.mFR, sdFR=self.decoder.sdFR, binlen=self.decoder.binlen, tslice=self.decoder.tslice)

        #Replace dtype as it was before running KF functions
        self.dtype = dtype_new

        # Compute sufficient stats for C and Q matrices (used for RML CLDA)
        n_features, n_states = C.shape
        R = np.mat(np.zeros([n_states, n_states]))
        S = np.mat(np.zeros([n_features, n_states]))
        R_small, S_small, T, ESS = clda.KFRML.compute_suff_stats(self.init_kin_features[self.ssm.train_inds, :], self.init_neural_features)

        R[np.ix_(self.ssm.drives_obs_inds, self.ssm.drives_obs_inds)] = R_small
        S[:,self.ssm.drives_obs_inds] = S_small

        self.decoder.filt.R = R
        self.decoder.filt.S = S
        self.decoder.filt.T = T
        self.decoder.filt.ESS = ESS
        self.decoder.n_features = n_features


    def load_decoder(self):
        ''' Overwritten in child classes '''
        pass


class SimKFDecoderSup(SimKFDecoder):
    '''
    Construct a KFDecoder based on encoder output in response to states simulated according to the state space model's process noise
    '''
    def load_decoder(self):
        '''
        Instantiate the neural encoder and "train" the decoder
        '''
        
        if hasattr(self, 'decoder'):
            print('Already have a decoder!')
        else:
            print("Creating simulation decoder..")
            print(self.encoder, type(self.encoder))
            n_samples = 2000
            units = self.encoder.get_units()
            n_units = len(units)
            print('units: ', n_units)

            # draw samples from the W distribution
            ssm = self.ssm
            A, _, W = ssm.get_ssm_matrices()
            mean = np.zeros(A.shape[0])
            mean[-1] = 1
            state_samples = np.random.multivariate_normal(mean, W, n_samples)

            spike_counts = np.zeros([n_units, n_samples])
            self.encoder.call_ds_rate = 1
            for k in range(n_samples):
                spike_counts[:,k] = np.array(self.encoder(state_samples[k], mode='counts')).ravel()

            kin = state_samples.T
            zscore = False
            if hasattr(self, 'clda_adapt_mFR_stats'):
                if self.clda_adapt_mFR_stats:
                    zscore = True
            print(' zscore decoder ? : ', zscore)
            self.decoder = train.train_KFDecoder_abstract(ssm, kin, spike_counts, units, 0.1, zscore=zscore)
            self.encoder.call_ds_rate = 6

            self.init_neural_features = spike_counts
            self.init_kin_features = kin

            super(SimKFDecoderSup, self).load_decoder()



class SimKFDecoderShuffled(SimKFDecoder):
    '''
    Construct a KFDecoder based on encoder output in response to states simulated according to the state space model's process noise
    '''
    def load_decoder(self):
        '''
        Instantiate the neural encoder and "train" the decoder
        '''
        print("Creating simulation decoder..")
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

        self.init_neural_features = spike_counts
        self.init_kin_features = kin

        self.decoder = train.train_KFDecoder_abstract(ssm, kin, spike_counts, units, 0.1)
        self.encoder.call_ds_rate = 6
        super(SimKFDecoderShuffled, self).load_decoder()

class SimKFDecoderRandom(SimKFDecoder):
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
