'''
Classes for BMI decoding using linear scaling. 
'''
import numpy as np

class State(object):
    '''For compatibility with other BMI decoding implementations'''

    def __init__(self, mean, *args, **kwargs):
        self.mean = mean

class LinearScaleFilter(object):

    model_attrs = ['attr']
    attrs_to_pickle = ['attr', 'obs', 'map']

    def __init__(self, n_counts, n_states, n_units, map=None, window=1, plant_gain=20):
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
        map : Which units to assign to which states (default = None)
            Floating point matrix of size (S, D) where S is the number of 
            states and D is the number of units, assigning a weight to each pair
        window : How many observations to average to smooth output (default = 1)
        plant_gain : how big is the screen, basically (default = 20)
            Maps from normalized output (0,1) to plant coordinates

        Returns
        -------
        LinearScaleFilter instance
        '''
        self.obs = np.zeros((n_counts, n_units))
        self.n_states = n_states
        self.n_units = n_units
        self.window = window
        self.map = map
        self.plant_gain = plant_gain
        if map is None:
            # Generate a default mapping where one unit controls one state
            self.map = np.identity(max(n_states, n_units))
            self.map = np.resize(self.map, (n_states, n_units))
        self.count = 0
        self.attr = dict(
            neural_mean = np.zeros(n_units),
            neural_std = np.ones(n_units),
            scaling_mean = np.zeros(n_units),
            scaling_std = np.ones(n_units),
        )
        self.fixed = False
        self._init_state()

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def __call__(self, obs, **kwargs):                                              # TODO need to pick single frequency band if given more than one
        self._add_obs(obs, **kwargs)
        if not self.fixed:
            self._update_scale_attr()
        self._init_state()

    def update_norm_attr(self, neural_mean=None, neural_std=None, scaling_mean=None, scaling_std=None):
        ''' Public method to set mean and std attributes'''
        if neural_mean is not None:
            self.attr.update(neural_mean = neural_mean)
        if neural_std is not None:
            self.attr.update(neural_std = neural_std)
        if scaling_mean is not None:
            self.attr.update(scaling_mean = scaling_mean)
        if scaling_std is not None:
            self.attr.update(scaling_std = scaling_std)

    def fix_norm_attr(self):
        ''' Stop fliter from self updating its attributes'''
        self.fixed = True
        
    def _pickle_init(self):
        self.fixed = True

    def _init_state(self):
        out = self._scale()
        self.state = State(out)

    def _add_obs(self, obs,**kwargs):
        ''' Normalize new observations and add them to the observation matrix'''

        # Z-score neural data
        norm_obs = (np.squeeze(obs) - self.attr['neural_mean']) / self.attr['neural_std']
        
        # Update observation matrix
        if self.count < len(self.obs): 
            self.count += 1
        self.obs[:-1, :] = self.obs[1:, :]
        self.obs[-1, :] = norm_obs
        
    def _scale(self):
        ''' Scale the (normalized) observations within the window'''

        # Normalize windowed average to 'scaling' mean and range
        if self.count == 0:
            m_win = np.zeros(np.size(self.obs, axis=1))
        elif self.count < self.window:
            m_win = np.squeeze(np.mean(self.obs[-self.count:, :], axis=0))
        else:
            m_win = np.squeeze(np.mean(self.obs[-self.window:, :], axis=0))
        x = (m_win - self.attr['scaling_mean']) / self.attr['scaling_std']
        
        # Arrange output according to map
        out = np.matmul(self.map, x).reshape(-1,1) * self.plant_gain
        return out

    def _update_scale_attr(self):
        ''' Update the normalization parameters'''

        # Normalize latest observation(s)
        mean = np.median(self.obs[-self.count:, :], axis=0)
        # range = max(1, np.amax(self.obs[-self.count:, :]) - np.amin(self.obs[-self.count:, :]))
        std = np.std(self.obs[-self.count:, :], axis=0)
        std[std == 0] = 1 # Hopefully this never happens
        self.update_norm_attr(scaling_mean=mean, scaling_std=std)


class PosVelState(State):
    ''' Simple state with the ability to integrate velocity over time'''

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
    ''' Linear filter that holds a position and velocity state'''

    def __init__(self, vel_control, *args, **kwargs):
        self.call_rate = kwargs.pop('call_rate')
        self.vel_control = vel_control
        super(PosVelScaleFilter, self).__init__(*args, **kwargs)

    def _init_state(self):
        self.state = PosVelState(self.vel_control, self.call_rate)
        out = self._scale()
        self.state.update(out)        

    def __call__(self, obs, **kwargs):
        self._add_obs(obs, **kwargs)
        if not self.fixed:
            self._update_scale_attr()
        out = self._scale()
        self.state.update(out)