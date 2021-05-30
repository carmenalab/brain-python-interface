
import numpy as np
from ..bmi import clda
from ..bmi import train

kinarm_bands = []
for i in np.arange(0,100,10):
    kinarm_bands.extend([[i, i+10]])
kinarm_bands.extend([[25, 40],[40, 55], [65, 90], [2, 100]])

class StateHolder(object):
    def __init__(self, x_mat, A_mat, powercap_flag, zbound, *args, **kwargs):
        if powercap_flag:
            pos_mean = np.sum((np.multiply(x_mat, A_mat)*0)+zbound[0], axis=0)
            self.mean = np.squeeze(np.hstack((np.array([pos_mean]), np.zeros((1,4)))))
        else:
            #self.mean = np.dot(x_array, A_array)
            pos_mean = np.sum(np.multiply(x_mat, A_mat), axis=0)
            self.mean = np.squeeze(np.hstack((np.array([pos_mean]), np.zeros((1,4)))))

class SmoothFilter(object):
    '''Moving Avergae Filter used in 1D or 2D LFP control:
    x_{t} = a0*x_{t} + a1*x_{t-1} + a2*x_{t-2} + ...

    Parameters

    ----------
    A: np.array of shape (N, 3)
        Weights for previous states
    X: np. array of previous states (N, 3)
    '''

    def __init__(self, n_steps, **kwargs):
        self.n_steps = n_steps
        A = np.ones(( self.n_steps, ))/float(self.n_steps)
        self.A = np.tile(np.array([A]).T, [1,3])

        self.control_method = 'fraction'
        self.current_lfp_pos = 0
        self.current_powercap_flag = 0

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def init_from_task(self,**kwargs):
        if 'n_steps' in kwargs:
            self.n_steps = kwargs['n_steps']
            A = np.ones(( self.n_steps, ))/float(self.n_steps)
            self.A = np.tile(np.array([A]).T, [1,3])

        if 'powercap' in kwargs:
            self.powercap = kwargs['powercap']

        if 'zboundaries' in kwargs:
            self.zboundaries = kwargs['zboundaries']

        if 'lfp_frac_lims' in kwargs:
            self.frac_lims = kwargs['lfp_frac_lims']

        if 'xlfp_frac_lims' in kwargs:
            self.xfrac_lims = kwargs['xlfp_frac_lims']

    def _init_state(self, init_state=None,**kwargs):
        if init_state is None:
            self.X = np.zeros(( self.n_steps, 3))

        #Implemented later: 
        elif init_state is 'average':
            if self.control_method == 'fraction':
                mn = np.mean(np.array(kwargs['frac_lim']))
            elif self.control_method == 'power':
                mn = np.mean(np.array(kwargs['pwr_mean']))
            self.X = np.zeros(( self.n_steps, 3 )) + mn

        self.state = StateHolder(self.X, self.A, 0, 0)
        
        self.cnt = 0

    def __call__(self, obs, **kwargs):
        self.state = self._mov_avg(obs, **kwargs)

    def _mov_avg(self, obs,**kwargs):
        #self.zboundaries = kwargs['zboundaries']
        self.fft_inds = kwargs['fft_inds']
        obs = obs.reshape(len(kwargs['channels']), len(kwargs['fft_freqs']))
        
        self.current_lfp_pos, self.current_powercap_flag = self.get_lfp_cursor(obs)
        
        self.X = np.vstack(( self.X[1:,:], self.current_lfp_pos ))
        return StateHolder(self.X, self.A, self.current_powercap_flag, self.zboundaries)

    def get_lfp_cursor(self, psd_est):
        'Obs: channels x frequencies '
        # Control band: 
        c_idx = self.control_band_ind

        #As done in kinarm script, sum together frequencies within a band, then take the mean across channels
        c_val = np.mean(np.sum(psd_est[:, self.fft_inds[c_idx]], axis=1))

        xc_idx = self.x_control_band_ind
        xc_val = np.mean(np.sum(psd_est[:, self.fft_inds[xc_idx]], axis=1))

        p_idx = self.totalpw_band_ind
        p_val = np.mean(np.sum(psd_est[:, self.fft_inds[p_idx]], axis=1))

        if self.control_method == 'fraction':
            lfp_control = c_val / float(p_val)
            xlfp_control = xc_val/float(p_val)

        elif self.control_method == 'power':
            lfp_control = c_val

        cursor_pos = self.lfp_to_cursor(lfp_control, xlfp_control)

        #if p_val <= self.powercap:
        #write c_val, xc_val, p_val, to file:
        if self.cnt < 3000:
            self.files[0].write(str(c_val)+',')
            self.files[1].write(str(p_val)+',')
            self.files[2].write(str(xc_val)+',')
            self.cnt += 1
        elif self.cnt == 3000:
            for f in self.files:
                f.close()
                
        #Hack: make x axis control powercap value:
        if xc_val <= self.powercap:
            powercap_flag = 0
        else:
            powercap_flag = 1

        return cursor_pos, powercap_flag

    def lfp_to_cursor(self, lfppos, xlfp_control):
        if self.control_method == 'fraction':
            dmn = lfppos - np.mean(self.frac_lims);
            cursor_pos = dmn * (self.zboundaries[1]-self.zboundaries[0]) / (self.frac_lims[1] - self.frac_lims[0])
            
            #Xcursor postion, keep within (0, -16) boundaries: 
            xdmn = xlfp_control - np.mean(self.xfrac_lims)
            xcursor_pos = xdmn * (0--16) / (self.xfrac_lims[1] - self.xfrac_lims[0])
            xcursor_pos = xcursor_pos - 8; #nonzero offset
            return np.array([xcursor_pos, 0, cursor_pos])


    def _pickle_init(self):
        pass

from .bmi import Decoder
class One_Dim_LFP_Decoder(Decoder):

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


    def init_from_task(self,**kwargs):
        if 'lfp_control_band' in kwargs:
            self.filt.control_band_ind, self.extractor_kwargs['bands'], self.extractor_kwargs['fft_inds'] = \
            self._get_band_ind(self.extractor_kwargs['fft_freqs'], kwargs['lfp_control_band'], self.extractor_kwargs['bands'])


        if 'lfp_totalpw_band' in kwargs:
            self.filt.totalpw_band_ind, self.extractor_kwargs['bands'], self.extractor_kwargs['fft_inds'] = \
            self._get_band_ind(self.extractor_kwargs['fft_freqs'], kwargs['lfp_totalpw_band'], self.extractor_kwargs['bands'])

        if 'xlfp_control_band' in kwargs:
            self.filt.x_control_band_ind, self.extractor_kwargs['bands'], self.extractor_kwargs['fft_inds'] = \
            self._get_band_ind(self.extractor_kwargs['fft_freqs'], kwargs['xlfp_control_band'], self.extractor_kwargs['bands'])

        c_txt = open('/home/helene/Downloads/pk/txt_write/control.txt','w')
        p_txt = open('/home/helene/Downloads/pk/txt_write/tot.txt','w')
        xc_txt = open('/home/helene/Downloads/pk/txt_write/x_cont.txt','w')
        self.filt.files = [c_txt, p_txt, xc_txt]

    def _get_band_ind(self, freq_pts, band, band_set):
        band_ind = -1
        for b, bd in enumerate(band_set):
            if (bd[0]==band[0]) and (bd[1]==band[1]):
                band_ind = b
        if band_ind == -1:
            band_ind = b+1
            band_set.extend([band])

        fft_ind = dict()
        for band_idx, band in enumerate(band_set):
            fft_ind[band_idx] = [freq_idx for freq_idx, freq in enumerate(freq_pts) if band[0] <= freq < band[1]]

        return band_ind, band_set, fft_ind

    
def _init_decoder_for_sim(n_steps = 10):
    kw = dict(control_method='fraction')
    sf = SmoothFilter(n_steps,**kw)
    ssm = train.endpt_2D_state_space
    units = [[23, 1],[24,1],[25,1]]
    decoder = One_Dim_LFP_Decoder(sf, units, ssm, binlen=0.1, n_subbins=1)
    learner = clda.DumbLearner()
        
    return decoder

def create_decoder(units, ssm, extractor_cls, extractor_kwargs, n_steps=10):
    kw = dict(control_method='fraction')
    sf = SmoothFilter(n_steps,**kw)
    decoder = One_Dim_LFP_Decoder(sf, units, ssm, extractor_cls, extractor_kwargs)

    return decoder





