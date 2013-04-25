import numpy as np
from . import VelocityBMI, ManualBMI

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

class KalmanFilter(VelocityBMI, ManualBMI):
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
        assert kindata.shape[1] == neuraldata.shape[1]

        self.neuraldata = neuraldata
        self.kindata = kindata

        if isinstance(kindata, np.ma.core.MaskedArray):
            mask = ~kindata.mask[0,:] # NOTE THE INVERTER 
            inds = np.nonzero([ mask[k]*mask[k+1] for k in range(len(mask)-1)])[0] 

            X = np.mat(kindata[:,mask])
            T = len(np.nonzero(mask)[0])

            Y = np.mat(neuraldata[:,mask])
            X_1 = np.mat(kindata[:, inds])
            X_2 = np.mat(kindata[:, inds+1])
            if include_offset: 
                X = np.vstack([ X, np.ones([1,T]) ])
                X_1 = np.vstack([ X_1, np.ones([1, len(inds)])])
                X_2 = np.vstack([ X_2, np.ones([1, len(inds)])])

        else:
            num_kindata, T = kindata.shape
            X = np.mat(kindata)
            if include_offset: 
                X = np.vstack([ X, np.ones([1,T]) ])
            Y = np.mat(neuraldata)
    
            # ML estimate of A and W
            X_1 = X[:,:-1]
            X_2 = X[:,1:]
    
        # ML estimate of C and Q
        C = Y*np.linalg.pinv(X)
        Q = np.cov( Y-C*X, bias=1 )
            
        A = X_2*np.linalg.pinv(X_1)
        W = np.cov(X_2 - A*X_1, bias=1)

        def _gen_A(t, s, m, n, off, ndim=3):
            A = np.zeros([2*ndim+1, 2*ndim+1]) # TODO remove hardcoding
            A_lower_dim = np.array([[t, s], [m, n]])
            A[0:2*ndim, 0:2*ndim] = np.kron(A_lower_dim, np.eye(ndim))
            return np.mat(A)

        T_per = 0.1 # TODO should this be 1./60?
        m = 0 
        a = 0.8
        w = 500
        A = _gen_A(1, T_per, m, a, 1)
        W = _gen_A(0, 0, 0, w, 0)
   
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

    def __call__(self, obs_t, **kwargs):
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
        return self.predict(super(KalmanFilter, self).__call__(obs_t), **kwargs)

    def predict(self, obs_t, **kwargs):
        obs_t = np.mat(obs_t).T
        pred_state = self.A*self.state + self.state_noise
        pred_obs = self.C*pred_state + self.obs_noise 

        C, Q = self.C, self.Q
        P_prior = pred_state.cov
        K = P_prior*C.T*np.linalg.pinv( C*P_prior*C.T + Q )

        pred_state.mean += K*(obs_t - pred_obs.mean)
        pred_state.cov = (self.I_N - K*C)*P_prior 

        self.state = pred_state
        return np.array(self.state.mean[:-1]).ravel() 

    def __setstate__(self, state):
        super(KalmanFilter, self).__setstate__(state)
        self.init_state()

class KalmanAssist(KalmanFilter):
    def predict(self, obs_t, target=None, assist_level=0.01):
        cursorpos = super(KalmanAssist, self).predict(obs_t)[:3]
        if target == None: # make sure target is a 1D array
            return cursorpos
        else:
            assert isinstance(target, np.ndarray)
            assert len(target.shape) == 1

            # TODO remove!
            try:
                assist_level = float(open('/home/helene/constants/assist_level', 'r').readline().rstrip())
            except:
                assist_level = 0.01

            oracle_vec_to_targ = target - np.array(self.state.mean[0:2,:]).ravel()
            max_speed = 0.02 # cm/sec
            time_to_targ = (0.1*oracle_vel_to_targ)/max_speed # time to target in sec, based on max speed 
            dt = 1./60
            num_iter_to_targ = time_to_targ / dt
            oracle_vel = 1./num_iter_to_targ * oracle_vec_to_target

            assisted_output = assist_level*oracle_vel + (1-assist_level)*cursorpos

            # modify state of the BMI 
            self.state.mean[0:3] = assisted_output
