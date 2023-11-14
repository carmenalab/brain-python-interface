'''
Classes for BMI decoding using linear scaling. 
'''
import numpy as np
from ..bmi.bmi import Filter

class State(object):
    '''For compatibility with other BMI decoding implementations'''

    def __init__(self, mean, *args, **kwargs):
        self.mean = mean

class LinearScaleFilter(Filter):

    model_attrs = ['attr']
    attrs_to_pickle = ['attr', 'obs', 'unit_to_state']

    def __init__(self, n_counts, n_states, n_units, unit_to_state=None, smoothing_window=1, decoder_to_plant=20, reject_threshold=5):
        '''
        Constructor for LinearScaleFilter

        Parameters
        ----------
        n_counts : Number of observations to hold
            Range is computed over the whole observation matrix size (N, D) 
            where N is the number of observations and D is the number of units
        n_states : How many state variables are there
            For example, a one-dim decoder has one state variable
        n_units : Number of neural units
            Can be number of isolated spiking units or number of channels for lfp
        unit_to_state : Which units to assign to which states (default = None)
            Floating point matrix of size (S, D) where S is the number of 
            states and D is the number of units, assigning a weight to each pair
        smoothing_window : How many observations to average to smooth output (default = 1)
        decoder_to_plant : how big is the screen, basically (default = 20)
            Maps from normalized output (0,1) to plant coordinates
        reject_threshold : How many standard deviations to reject
            If the standard deviation of the latest observation is greater than this,
            the filter will reject the observation
            
        Returns
        -------
        LinearScaleFilter instance
        '''
        self.obs = np.zeros((n_counts, n_units))
        self.n_states = n_states
        self.n_units = n_units
        self.smoothing_window = smoothing_window
        self.unit_to_state = unit_to_state
        self.decoder_to_plant = decoder_to_plant
        if unit_to_state is None:
            # Generate a default mapping where one unit controls one state
            self.unit_to_state = np.identity(max(n_states, n_units))
            self.unit_to_state = np.resize(self.unit_to_state, (n_states, n_units))
        self.count = 0
        self.attr = dict(
            neural_mean = np.zeros(n_units),
            neural_std = np.ones(n_units),
            offset = np.zeros(n_units),
            scale = np.ones(n_units),
        )
        self.fixed = False
        self.reject_threshold = reject_threshold
        self._init_state()

    def get_mean(self):
        ''' Must return self.state.mean to maintain compatibility'''
        return np.array(self.state.mean).ravel()

    def __call__(self, obs, **kwargs):
        ''' 
        Update the state with the new neural observation

        Parameters
        ----------
        obs : Neural observations (1 x P*N)
            where P is the number of features per channel and N is the number of channels
        kwargs : For consistency with other BMI decoders

        Returns
        ----------
        '''     

        # Z-score neural data
        norm_obs = (np.squeeze(obs) - self.attr['neural_mean']) / self.attr['neural_std']

        # Reject if the standard deviation of the latest observation is too high
        norm_obs[abs(norm_obs) > self.reject_threshold] = np.nan

        # Add observation
        self._add_obs(norm_obs, **kwargs)
        if not self.fixed:
            self._update_scale_attr()
        self._update_state()

    def update_norm_attr(self, neural_mean=None, neural_std=None, offset=None, scale=None):
        ''' Public method to set mean and std attributes'''
        if neural_mean is not None:
            self.attr.update(neural_mean = neural_mean)
        if neural_std is not None:
            self.attr.update(neural_std = neural_std)
        if offset is not None:
            self.attr.update(offset = offset)
        if scale is not None:
            self.attr.update(scale = scale)

    def fix_norm_attr(self):
        ''' Stop fliter from self updating its attributes'''
        self.fixed = True

    def unfix_norm_attr(self):
        ''' Let fliter update its scale attributes'''
        self.fixed = False

    def _pickle_init(self):
        #self.fix_norm_attr()
        pass

    def _init_state(self):
        ''' Required by decoder'''
        out = self._scale()
        self.state = State(out)

    def _update_state(self):
        out = self._scale()
        self.state = State(out)

    def _add_obs(self, obs,**kwargs):
        ''' Add new observations to the observation matrix'''
        
        if np.any(np.isnan(obs)):
            return

        # Update observation matrix
        if self.count < len(self.obs): 
            self.count += 1
        self.obs[:-1, :] = self.obs[1:, :]
        self.obs[-1, :] = obs
        
    def _get_latest_obs(self):
        ''' Smooth observations by taking the average across the smoothing window'''
        if self.count == 0:
            mean = np.zeros(np.size(self.obs, axis=1))
        elif self.count < self.smoothing_window:
            mean = np.squeeze(np. nanmean(self.obs[-self.count:, :], axis=0))
        else:
            mean = np.squeeze(np.nanmean(self.obs[-self.smoothing_window:, :], axis=0))
        return mean
        
    def _scale(self):
        ''' Scale the (normalized) observations '''

        # Normalize windowed average to the current offset and scale
        x = (self._get_latest_obs() - self.attr['offset']) / self.attr['scale']
        
        # Arrange output according to map
        out = np.matmul(self.unit_to_state, x).reshape(-1,1) * self.decoder_to_plant
        return out

    def _update_scale_attr(self):
        ''' Update the normalization parameters'''

        # Normalize latest observation(s)
        mean = np.nanmedian(self.obs[-self.count:, :], axis=0)
        # range = max(1, np.amax(self.obs[-self.count:, :]) - np.amin(self.obs[-self.count:, :]))
        std = np.nanstd(self.obs[-self.count:, :], axis=0)
        std[std < 1e-6] = 1e-6 # Avoid divide by zero
        self.update_norm_attr(offset=mean, scale=std)


class PosVelState(State):
    ''' 
    Simple state with the ability to integrate velocity over time
    Only compatible with StateSpaceEndptVel2D state space matrices
    '''

    def __init__(self, vel_control, call_rate=60):
        self.vel_control = vel_control
        self.call_rate = call_rate
        self.mean = np.zeros((7,1))

    def update(self, mean):
        if self.vel_control:
            self.mean[3:6] = mean[3:6]

            # Add the velocity (units/s) to the position (units)
            self.mean[0:3] = self.mean[3:6] / self.call_rate + self.mean[0:3]
        else:
            self.mean = mean

class PosVelScaleFilter(LinearScaleFilter):
    ''' 
    Linear filter that holds a position and velocity state
    Only compatible with StateSpaceEndptVel2D state space matrices
    '''

    def __init__(self, vel_control, *args, **kwargs):
        self.call_rate = kwargs.pop('call_rate')
        self.vel_control = vel_control
        super(PosVelScaleFilter, self).__init__(*args, **kwargs)

    def _init_state(self):
        self.state = PosVelState(self.vel_control, self.call_rate)
        out = self._scale()
        self.state.update(out)        

    def _update_state(self):
        out = self._scale()
        self.state.update(out)  

def create_lindecoder(ssm, units, unit_to_state, decoder_to_plant=None, smoothing_window=1, vel_control=False, update_rate=0.1):
    from riglib.bmi import Decoder
    filt_counts = smoothing_window # only used for smoothing since we're fixing the gains
    filt = PosVelScaleFilter(vel_control, filt_counts, ssm.n_states, len(units), unit_to_state=unit_to_state, smoothing_window=smoothing_window, decoder_to_plant=decoder_to_plant, call_rate=1/update_rate)
    
    # calculate gains from training data
    # filt.update_norm_attr(neural_mean=0, neural_std=1, offset=mFR, scale=sdFR)
    
    decoder = Decoder(filt, units, ssm, binlen=update_rate, subbins=1)
    decoder.n_features = len(units)
    decoder.binlen = update_rate

    return decoder