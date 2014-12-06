from riglib import bmi
from riglib.bmi import extractor
import numpy as np
from riglib.bmi import clda
from riglib.bmi import train

kinarm_bands = []
for i in np.arange(0,100,10):
    kinarm_bands.extend([[i, i+10]])
kinarm_bands.extend([[25, 40],[40, 55], [65, 90], [2, 100]])

class StateHolder(object):
    def __init__(self, x_array, A_array, *args, **kwargs):
        self.mean = np.dot(x_array, A_array)

class SmoothFilter(StateHolder):
    '''Moving Avergae Filter used in 1D LFP control:
    x_{t} = a0*x_{t} + a1*x_{t-1} + a2*x_{t-2} + ...

    Parameters
    ----------
    A: np.array of shape (N, )
        Weights for previous states
    X: np. array of previous states (N, )
    '''
    model_attrs = []

    def __init__(self, n_steps, **kwargs):
        self.n_steps = n_steps
        self.A = np.ones(( n_steps, ))/float(n_steps)
        
    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def _init_state(self, init_state=None,**kwargs):
        if init_state is None:
            self.X = np.zeros(( self.n_steps, ))

        elif init_state is 'average':
            if control_method == 'fraction':
                mn = np.mean(np.array(kwargs['frac_lim']))
            elif control_method == 'total_power':
                mn = np.mean(np.array(kwargs['pwr_mean']))
            self.X = np.zeros(( self.n_steps )) + mn

        self.state = StateHolder(self.X, self.A)

    def __call__(self, obs, **kwargs):
        self.state = self._mov_avg(obs, **kwargs)

    def _mov_avg(self, obs):
        print stop
        self.X = np.hstack(( self.X[1:], obs ))
        return DummyState(self.X, self.A)

class One_Dim_LFP_Decoder(bmi.Decoder):

    def __init__(self, *args, **kwargs):
        
        bands = kinarm_bands
        control_method='fraction'
        no_log=True
        
        super(One_Dim_LFP_Decoder, self).__init__(*args, **kwargs)
        
        if no_log:
            kw = dict(no_log=no_log)

        #For now: 
        source = None
        self.extractor_cls = extractor.LFPMTMPowerExtractor(source,self.units,bands=bands,**kw)
        self.extractor_kwargs = self.extractor_cls.extractor_kwargs

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self,key,value)

    def predict(self, neural_obs, **kwargs):
        self.filt(neural_obs, **kwargs)
    

def _init_decoder_for_sim(n_steps = 10):
    kw = dict(control_method='fraction')
    sf = SmoothFilter(n_steps,**kw)
    ssm = train.endpt_2D_state_space
    units = [[23, 1],[24,1],[25,1]]
    decoder = One_Dim_LFP_Decoder(sf, units, ssm, binlen=0.1, n_subbins=1)

    extractor_cls = decoder.extractor_cls
    extractor_kwargs = decoder.extractor_kwargs
    decoder.n_features = len(extractor_kwargs['bands'])*len(extractor_kwargs['channels'])

    learner = clda.DumbLearner()
    bmi_system = bmi.BMISystem(decoder, learner, None)

    #Get neural observations
    #Choose frequency band
    #Average across channels: 

    lfp_power = np.random.randn(decoder.n_features, 1)

    feature_type=extractor_cls.feature_type
    target_state = np.zeros([decoder.n_states, decoder.n_subbins])
 

    decoder_output, update_flag = bmi_system(lfp_power, target_state, 'target', feature_type = feature_type)
        
    return decoder, learner, None






