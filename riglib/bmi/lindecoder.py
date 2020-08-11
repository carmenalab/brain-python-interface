'''
Classes for BMI decoding using linear scaling. 
'''
import numpy as np

class State(object):
    '''For compatibility with other BMI decoding implementations, literally just holds the state'''

    def __init__(self, mean, *args, **kwargs):
        self.mean = mean

class LinearScaleFilter(object):

    def __init__(self, n_counts, window, n_states, n_units):
        '''
        Parameters:

        n_counts How many observations to hold
        window   How many observations to average
        n_states How many state space variables are there
        n_units  Number of neural units
        '''
        self.state = State(np.zeros([n_states,1]))
        self.obs = np.zeros((n_counts, n_units))
        self.n_states = n_states
        self.window = window
        self.n_units = n_units
        self.count = 0

    def _init_state(self):
        pass

    def get_mean(self):
        return np.array(self.state.mean).ravel()

    def __call__(self, obs, **kwargs):
        self.state = self._normalize(obs, **kwargs)

    def _normalize(self, obs,**kwargs):
        ''' Function to compute normalized scaling of new observations'''

        self.obs[:-1, :] = self.obs[1:, :]
        self.obs[-1, :] = np.squeeze(obs)
        if self.count < len(self.obs): 
            self.count += 1

        m_win = np.squeeze(np.mean(self.obs[-self.window:, :], axis=0))
        m = np.median(self.obs[-self.count:, :], axis=0)
        # range = max(1, np.amax(self.obs[-self.count:, :]) - np.amin(self.obs[-self.count:, :]))
        range = np.std(self.obs[-self.count:, :], axis=0)*3
        range[range < 1] = 1
        x = (m_win - m) / range
        x = np.squeeze(np.asarray(x)) * 20 # hack for 14x14 cursor
        
        # Arrange output
        if self.n_states == self.n_units:
            return State(x)
        elif self.n_states == 3 and self.n_units == 2:
            mean = np.zeros([self.n_states,1])
            mean[0] = x[0]
            mean[2] = x[1]
            return State(mean)
        else:
            raise NotImplementedError()
    