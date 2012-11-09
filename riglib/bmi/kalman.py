import numpy as np
from . import VelocityBMI

class GaussianState(object):
    def __init__(self, mean, cov):
        if isinstance(mean, float):
            self.mean = mean
        else:
            self.mean = np.mat(mean.reshape(-1,1))
        self.cov = np.mat(cov)
    
    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, np.float64) or isinstance(other, float):
            mu = other*self.mean
            cov = other**2 * self.cov
        elif isinstance(other, np.matrix):
            mu = other*self.mean
            cov = other*self.cov*other.T
        elif isinstance(other, np.ndarray):
            other = mat(array)
            mu = other*self.mean
            cov = other*self.cov*other.T
        else:
            print type(other)
            raise
        return GaussianState(mu, cov)

    def __mul__(self, other):
        mean = other*self.mean
        if isinstance(other, int) or isinstance(other, np.float64) or isinstance(other, float):
            cov = other**2 * self.cov
        else:
            print type(other)
            raise
        return GaussianState(mean, cov)

    def __add__(self, other):
        if isinstance(other, GaussianState):
            return GaussianState( self.mean+other.mean, self.cov+other.cov )

class KalmanFilter(VelocityBMI):
    def __init__(self, *args, **kwargs):
        super(KalmanFilter, self).__init__(*args, **kwargs)
        kindata, neurons = self.get_data()
        kindata = kindata[:, [0, 8]].reshape(len(kindata), -1)
        self.train( kindata.T, neurons.T, include_offset=True)
        self.means = kindata.mean(0)
    
    def train(self, kindata, neuraldata, include_offset=True):
        """ Train a kalman filter
        
        Parameters
        ----------
        kindata : array-like
            2D array of kinematic variable observations, time flows on axis 1
        neuraldata : array-like
            2D array of spike counts, time flows on axis 1        
        include_offset : bool, optional, default=True
            If true, a state element that is always 1 for offset in ML fitting
        """
        self.include_offset = include_offset

        num_kindata, T = kindata.shape
        X = np.mat(kindata)
        if include_offset: 
            X = np.vstack([ X, np.ones([1,T]) ])
        Y = np.mat(neuraldata)
    
        # ML estimate of C and Q
        C = Y*np.linalg.pinv(X)
        Q = np.cov( Y-C*X, bias=1 )
        
        # ML estimate of A and W
        X_1 = X[:,:-1]
        X_2 = X[:,1:]
    
        if include_offset:
            A = np.mat(np.zeros([num_kindata+1, num_kindata+1]))
        else:
            A = np.mat(np.zeros([num_kindata, num_kindata]))
            
        A = X_2*np.linalg.pinv(X_1)
        W = np.cov(X_2 - A*X_1, bias=1)

        # TODO estimate ML diagonal A
        # TODO estimate ML diagonal W
        # TODO estimate Q with orth. constraints
    
        # store KF matrix parameters 
        self.A = A
        self.W = W
        self.C = C
        self.Q = Q

        self.obs_noise = GaussianState(0.0, Q)
        self.state_noise = GaussianState(0.0, W)
        self.I_N = np.mat(np.eye(self.C.shape[1]))

        self.init_state()

    def init_state(self):
        """
        Initialize the state of the KF 
        """
        ## Initialize the BMI state, assuming 
        nS = self.A.shape[0] # number of state variables
        init_state = np.mat( np.zeros([nS, 1]) )
        init_state[-1,0] = 1
        init_cov = np.mat( np.zeros([nS, nS]) )
        self.state = GaussianState(init_state, init_cov) 

    def __call__(self, obs_t):
        '''
        Return the predicted arm position given the new data.

        Parameters
        -----------
        newdata : array_like
	        Recent spike data for all units

        Returns
        -------
        output : array_like
	        Decoder output for each decoded parameter

        '''
        return self.predict(super(KalmanFilter, self).__call__(obs_t))

    def predict(self, obs_t):
        obs_t = np.mat(obs_t).T
        pred_state = self.A*self.state + self.state_noise
        pred_obs = self.C*pred_state + self.obs_noise 

        C, Q = self.C, self.Q
        P_prior = pred_state.cov
        K = P_prior*C.T*np.linalg.pinv( C*P_prior*C.T + Q )

        pred_state.mean += K*(obs_t - pred_obs.mean)
        pred_state.cov = (self.I_N - K*C)*P_prior 

        self.state = pred_state
        return np.array(self.state.mean[:-1]).ravel() + self.means

    def __setstate__(self, state):
        super(KalmanFilter, self).__setstate__(state)
        self.init_state()