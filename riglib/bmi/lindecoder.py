'''
Classes for BMI decoding using linear scaling. 
'''
import numpy as np

class State(object):
    '''For compatibility with other BMI decoding implementations, literally just holds the state'''

    def __init__(self, mean, *args, **kwargs):
        self.mean = mean

class LinearScaleFilter(object):

    def __init__(self, n_counts, n_states, n_units, map=None, window=1):
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

        Returns
        -------
        LinearScaleFilter instance
        '''
        self.state = State(np.zeros([n_states,1]))
        self.obs = np.zeros((n_counts, n_units))
        self.n_states = n_states
        self.n_units = n_units
        self.window = window
        self.map = map
        if map is None:
            # Generate a default mapping where one unit controls one state
            self.map = np.identity(max(n_states, n_units))
            self.map = np.resize(self.map, (n_states, n_units))
        self.count = 0
        self.params = dict(
            neural_mean = np.zeros(n_units),
            neural_range = np.ones(n_units),
            scaling_mean = np.zeros(n_units),
            scaling_range = np.ones(n_units),
        )
        self.fixed = False

    def _init_state(self):
        pass

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def __call__(self, obs, **kwargs):                                              # TODO need to pick single frequency band if given more than one
        self.state = self._normalize(obs, **kwargs)

    def _normalize(self, obs,**kwargs):
        ''' Function to compute normalized scaling of new observations'''

        # Update observation matrix
        norm_obs = (obs.ravel() - self.params['neural_mean']) / self.params['neural_range'] # center on zero
        self.obs[:-1, :] = self.obs[1:, :]
        self.obs[-1, :] = norm_obs
        if self.count < len(self.obs): 
            self.count += 1

        if not self.fixed:
            self._update_scale_param(obs)
        m_win = np.squeeze(np.mean(self.obs[-self.window:, :], axis=0))
        x = (m_win - self.params['scaling_mean']) * self.params['scaling_range']
        
        # Arrange output according to map
        out = np.matmul(self.map, x).reshape(-1,1)
        return State(out)

    def _update_scale_param(self, obs):
        ''' Function to update the normalization parameters'''

        # Normalize latest observation(s)
        mean = np.median(self.obs[-self.count:, :], axis=0)
        # range = max(1, np.amax(self.obs[-self.count:, :]) - np.amin(self.obs[-self.count:, :]))
        range = 3 * np.std(self.obs[-self.count:, :], axis=0)
        range[range < 1] = 1
        self.update_norm_param(scaling_mean=mean, scaling_range=range)

    def update_norm_param(self, neural_mean=None, neural_range=None, scaling_mean=None, scaling_range=None):
        if neural_mean is not None:
            self.params.update(neural_mean = neural_mean)
        if neural_range is not None:
            self.params.update(neural_range = neural_range)
        if scaling_mean is not None:
            self.params.update(scaling_mean = scaling_mean)
        if scaling_range is not None:
            self.params.update(scaling_range = scaling_range)

    def fix_norm_param(self):
        self.fixed = True

    def get_norm_param(self):
        return self.params