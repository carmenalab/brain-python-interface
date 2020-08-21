'''
Classes for BMI decoding using linear scaling. 
'''
import numpy as np

class State(object):
    '''For compatibility with other BMI decoding implementations, literally just holds the state'''

    def __init__(self, mean, *args, **kwargs):
        self.mean = mean

class LinearScaleFilter(object):

    def __init__(self, n_counts, n_states, n_units, map=None, window=1, gain=20):
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
            Sum along each row must equal 1.0
        window : How many observations to average to smooth output (default = 1)
        gain : How far to move the plant for a normalized output of 1.0 (default = 20)

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
            # Generate a default map where one unit controls one state
            self.map = np.identity(max(n_states, n_units))
            self.map = np.resize(self.map, (n_states, n_units))
        self.gain = gain
        self.count = 0
        self.fixed = False

    def _init_state(self):
        pass

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def __call__(self, obs, **kwargs):
        self.state = self._normalize(obs, **kwargs)

    def _normalize(self, obs,**kwargs):
        ''' Function to compute normalized scaling of new observations'''

        # Update observation matrix, unless it has been fixed
        if not self.fixed:
            self.obs[:-1, :] = self.obs[1:, :]
            self.obs[-1, :] = np.squeeze(obs)
            if self.count < len(self.obs): 
                self.count += 1

        # Normalize latest observation(s)
        m_win = np.squeeze(np.mean(self.obs[-self.window:, :], axis=0))
        m = np.median(self.obs[-self.count:, :], axis=0)
        # range = max(1, np.amax(self.obs[-self.count:, :]) - np.amin(self.obs[-self.count:, :]))
        range = 3 * np.std(self.obs[-self.count:, :], axis=0)
        range[range < 1] = 1
        x = (m_win - m) / range * self.gain
        
        # Arrange output according to map
        out = np.matmul(self.map, x).reshape(-1,1)
        return State(out)

    def save_obs(self):
        raise NotImplementedError()

    def fix_obs(self):
        self.fixed = True

    def load_and_fix_obs(self, file):
        raise NotImplementedError()
        self.count = len(self.obs)
        self.fix_obs()