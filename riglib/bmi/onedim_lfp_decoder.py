
from riglib import bmi, plexon, source
from riglib.bmi import extractor
import numpy as np
from riglib.bmi import clda
from riglib.bmi import train

kinarm_bands = []
for i in np.arange(0,100,10):
    kinarm_bands.extend([[i, i+10]])
kinarm_bands.extend([[25, 40],[40, 55], [65, 90], [2, 100]])

class StateHolder(object):
    def __init__(self, x_array, A_array, powercap_flag, zbound, *args, **kwargs):
        if powercap_flag:
            self.mean = zbound[0]
        else:
            self.mean = np.dot(x_array, A_array)

class SmoothFilter(object):
    '''Moving Avergae Filter used in 1D LFP control:
    x_{t} = a0*x_{t} + a1*x_{t-1} + a2*x_{t-2} + ...

    Parameters

    ----------
    A: np.array of shape (N, )
        Weights for previous states
    X: np. array of previous states (N, )
    '''

    def __init__(self, n_steps, **kwargs):
        self.n_steps = n_steps
        self.A = np.ones(( n_steps, ))/float(n_steps)
        self.control_method = 'fraction'
        self.current_lfp_pos = 0
        self.current_powercap_flag = 0

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def _init_state(self, init_state=None,**kwargs):
        if init_state is None:
            self.X = np.zeros(( self.n_steps, ))

        #Implemented later: 
        elif init_state is 'average':
            if self.control_method == 'fraction':
                mn = np.mean(np.array(kwargs['frac_lim']))
            elif self.control_method == 'power':
                mn = np.mean(np.array(kwargs['pwr_mean']))
            self.X = np.zeros(( self.n_steps )) + mn

        self.state = StateHolder(self.X, self.A, 0, 0)
        self.file = open("/home/helene/preeya/tot_pw.txt","w")
        self.cnt = 0

    def __call__(self, obs, **kwargs):
        self.state = self._mov_avg(obs, **kwargs)

    def _mov_avg(self, obs,**kwargs):
        #self.zboundaries = kwargs['zboundaries']
        self.fft_inds = kwargs['fft_inds']
        obs = obs.reshape(len(kwargs['channels']), len(kwargs['fft_freqs']))
        self.current_lfp_pos, self.current_powercap_flag = self.get_lfp_cursor(obs)
        
        self.X = np.hstack(( self.X[1:], self.current_lfp_pos))
        return StateHolder(self.X, self.A, self.current_powercap_flag, self.zboundaries)

    def get_lfp_cursor(self, psd_est):
        'Obs: channels x frequencies '
        # Control band: 
        c_idx = self.control_band_ind

        #As done in kinarm script, sum together frequencies within a band, then take the mean across channels
        c_val = np.mean(np.sum(psd_est[:, self.fft_inds[c_idx]], axis=1))

        p_idx = self.totalpw_band_ind
        p_val = np.mean(np.sum(psd_est[:, self.fft_inds[p_idx]], axis=1))

        if self.control_method == 'fraction':
            lfp_control = c_val / float(p_val)
        elif self.control_method == 'power':
            lfp_control = c_val

        cursor_pos = self.lfp_to_cursor(lfp_control)

        if p_val <= self.powercap:
            powercap_flag = 0
        else:
            powercap_flag = 1

        if self.cnt<1000:
            s = "%.20f\n" % p_val
            self.file.write(str(s))
            self.cnt += 1
        elif self.cnt == 1000:
            self.file.close()

        return cursor_pos, powercap_flag

    def lfp_to_cursor(self, lfppos):
        if self.control_method == 'fraction':
            #print 'SELF_FRAC LIMS', self.frac_lims
            dmn = lfppos - np.mean(self.frac_lims);
            cursor_pos = dmn * (self.zboundaries[1]-self.zboundaries[0]) / (self.frac_lims[1] - self.frac_lims[0])
            return cursor_pos


    def _pickle_init(self):
        pass


class One_Dim_LFP_Decoder(bmi.Decoder):

    def __init__(self, *args, **kwargs):
        
        #Args: sf, units, ssm, extractor_cls, extractor_kwargs
        super(One_Dim_LFP_Decoder, self).__init__(args[0], args[1], args[2])
        
        self.extractor_cls = args[3]
        self.extractor_kwargs = args[4]

        #For now, hardcoded: 
        bands = kinarm_bands
        control_method='fraction'
        no_log=True
        no_mean=True

        self.extractor_kwargs['bands'] = bands
        self.extractor_kwargs['no_log'] = no_log
        self.extractor_kwargs['no_mean'] = no_mean

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self,key,value)

    def predict(self, neural_obs, **kwargs):
        #kwargs['zboundaries'] = self.filt.zboundaries
        kwargs['fft_inds'] = self.extractor_kwargs['fft_inds']
        kwargs['channels'] = self.extractor_kwargs['channels']
        kwargs['fft_freqs'] = self.extractor_kwargs['fft_freqs']

        self.filt(neural_obs, **kwargs)
    
def _init_decoder_for_sim(n_steps = 10):
    kw = dict(control_method='fraction')
    sf = SmoothFilter(n_steps,**kw)
    ssm = train.endpt_2D_state_space
    units = [[23, 1],[24,1],[25,1]]
    decoder = One_Dim_LFP_Decoder(sf, units, ssm, binlen=0.1, n_subbins=1)
    learner = clda.DumbLearner()
        
    return decoder

def create_decoder(units, ssm, extractor_cls, extractor_kwargs, n_steps=5):
    print 'N_STEPS: ', n_steps
    kw = dict(control_method='fraction')
    sf = SmoothFilter(n_steps,**kw)
    decoder = One_Dim_LFP_Decoder(sf, units, ssm, extractor_cls, extractor_kwargs)

    return decoder





