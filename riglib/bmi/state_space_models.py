#!/usr/bin/python
"""
State-space models for different types of BMIs (i.e. representations of various 
BMI "plants"), and methods to manipulate the parameterizations of such models 
e.g. time resampling of a linear discrete-time representation of a continuous-time model). 
"""
import numpy as np

############
## Constants
############
pi = np.pi
deg_to_rad = pi/180

def resample_ssm(A, W, Delta_old=0.1, Delta_new=0.005, include_offset=True):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    A = A.copy()
    W = W.copy()
    if include_offset:
        orig_nS = A.shape[0]
        A = A[:-1, :-1]
        W = W[:-1, :-1]

    loop_ratio = Delta_new/Delta_old
    N = 1./loop_ratio
    A_new = A**loop_ratio
    nS = A.shape[0]
    I = np.mat(np.eye(nS))
    W_new = W * ( (I - A_new**N) * (I - A_new).I - I).I
    if include_offset:
        A_expand = np.mat(np.zeros([orig_nS, orig_nS]))
        A_expand[:-1,:-1] = A_new
        A_expand[-1,-1] = 1
        W_expand = np.mat(np.zeros([orig_nS, orig_nS]))
        W_expand[:-1,:-1] = W_new
        return A_expand, W_expand
    else:
        return A_new, W_new

def resample_scalar_ssm(a, w, Delta_old=0.1, Delta_new=0.005):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    loop_ratio = Delta_new/Delta_old
    a_delta_new = a**loop_ratio
    w_delta_new = w / ((1-a_delta_new**(2*(1./loop_ratio)))/(1- a_delta_new**2))

    mu = 1
    sigma = 0
    for k in range(int(1./loop_ratio)):
        mu = a_delta_new*mu
        sigma = a_delta_new * sigma * a_delta_new + w_delta_new
    return a_delta_new, w_delta_new

def _gen_A(t, s, m, n, off, ndim=3):
    """
    Utility function for generating block-diagonal matrices
    used by the KF
    
        [t*I, s*I, 0
         m*I, n*I, 0
         0,   0,   off]

    Parameters
    ----------

    Returns
    -------
    np.mat of shape (N, N); N = 2*ndim + 1
    """
    A = np.zeros([2*ndim+1, 2*ndim+1])
    A_lower_dim = np.array([[t, s], [m, n]])
    A[0:2*ndim, 0:2*ndim] = np.kron(A_lower_dim, np.eye(ndim))
    A[-1,-1] = off
    return np.mat(A)

def linear_kinarm_kf(update_rate=1./10, units_mult=0.01, ndim=3, vel_decay=0.8):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    Delta_KINARM = 1./10
    loop_update_ratio = update_rate/Delta_KINARM
    w_in_meters = 0.0007
    w_units_resc = w_in_meters / (units_mult ** 2)
    a_resampled, w_resampled = resample_scalar_ssm(vel_decay, w_units_resc, Delta_old=Delta_KINARM, Delta_new=update_rate)
    A = _gen_A(1, update_rate, 0, a_resampled, 1, ndim=ndim)
    W = _gen_A(0, 0, 0, w_resampled, 0, ndim=ndim)
    return A, W
    

class State(object):
    '''
    A 1D component of a state-space, e.g., vertical velocity
    '''
    def __init__(self, name, stochastic=False, drives_obs=False, min_val=np.nan, max_val=np.nan, order=-1):
        '''
        Constructor for State

        Parameters
        ----------
        name : string
            Name of the state
        stochastic : bool, optional
            Specify whether the state is stochastic (estimated from observation) 
            or deterministic (updated entirely by model). Default is 'deterministic'
        drives_obs : bool, optional
            Specify whether the state will be reflected in the observations if it is
            used as a 'hidden' state. By default, the state and any observations will not be directly related
        min_val : float, optional
            Hard (nonlinear) constraint on the minimum value of the state. By default (np.nan), no constraint is applied.
        max_val : float, optional
            Hard (nonlinear) constraint on the maximum value of the state. By default (np.nan), no constraint is applied.
        order : int
            Specification of the 'order' that this state would contribute to a differential equation, 
            e.g., constant states are order 0, position states are order 1, velocity states are order 2, etc.

        Returns
        -------
        State instance
        '''
        assert not name == 'q', "'q' is a reserved keyword (symbol for generalized robot coordinates) and cannot be used as a state name"
        self.name = name
        self.stochastic = stochastic 
        self.drives_obs = drives_obs
        self.min_val = min_val
        self.max_val = max_val
        self.order = order

    def __repr__(self):
        return str(self.name) 

    def __eq__(self, other):
        '''
        State instances are equal if all their attributes (name, stochastic, etc.) are equal
        '''
        if not isinstance(other, State):
            return False
        else:
            return np.all([self.__dict__[x] == other.__dict__[x] or (np.isnan(self.__dict__[x]) and np.isnan(other.__dict__[x])) for x in self.__dict__])

class StateSpace(object):
    '''
    A collection of multiple 'State' instances forms a StateSpace
    '''
    def __init__(self, *states):
        '''
        Constructor for StateSpace

        Parameters
        ----------
        states : packed tuple
            State instances specified in comma-separated arguments

        Returns
        -------
        StateSpace instance
        '''
        self.states = list(states)

    def __repr__(self):
        return 'State space: ' + str(self.state_names)

    @property
    def is_stochastic(self):
        '''
        An array of booleans specifying each state as stochastic
        '''
        return np.array([x.stochastic for x in self.states])

    @property
    def drives_obs(self):
        '''
        An array of booleans specifying each state as observation-driving
        '''
        return np.array([x.drives_obs for x in self.states])

    @property
    def state_names(self):
        '''
        A list of string names for each state
        '''
        return [x.name for x in self.states]

    @property
    def bounding_box(self):
        '''
        A tuple of min values and max values for each state
        '''
        min_bounds = np.array(filter(lambda x: x is not np.nan, [x.min_val for x in self.states]))
        max_bounds = np.array(filter(lambda x: x is not np.nan, [x.max_val for x in self.states]))
        return (min_bounds, max_bounds)

    @property
    def states_to_bound(self):
        '''
        A list of the names of all the states which have limits on the values they can take.
        '''
        return [x.name for x in filter(lambda x: x.min_val is not np.nan, self.states)]

    @property
    def n_states(self):
        '''
        Number of states in the space
        '''
        return len(self.states)

    @property
    def train_inds(self):
        '''
        An array of 
        '''
        return filter(lambda k: self.states[k].stochastic, range(self.n_states))

    @property
    def drives_obs_inds(self):
        '''
        A list of the indices of the states which are related to observations when 
        used as a hidden state-space. Used when seeding Decoders
        '''
        return filter(lambda k: self.states[k].drives_obs, range(self.n_states))

    @property 
    def state_order(self):
        '''
        An array listing the 'order' of each state (see State.__init__ for description of 'order')
        '''
        return np.array([x.order for x in self.states])

    def get_ssm_matrices(self, *args, **kwargs):
        '''
        Returns the parameters of the composite state-space models for use in Decoders. 
        Must be overridden in child classes as there is no way to specify this generically.
        '''
        raise NotImplementedError

    def __eq__(self, other):
        '''
        State spaces are equal if the all the states are equal and all the states are listed in the same order
        '''
        if not isinstance(other, StateSpace):
            return False
        else:
            return self.states == other.states

class LinearVelocityStateSpace(StateSpace):
    def __init__(self, states, vel_decay=0.8, w=7, Delta=0.1):
        self.states = states 
        self.vel_decay = vel_decay
        self.w = w
        self.Delta = Delta

        # check that there are an equal number of pos and vel states
        assert len(np.nonzero(self.state_order == 0)[0]) == len(np.nonzero(self.state_order == 1)[0])

    def __setstate__(self, state):
        self.__dict__ = state
        if not hasattr(self, 'Delta'):
            self.Delta = 0.1

        if not hasattr(self, 'vel_decay'):
            self.vel_decay = 0.8

        if not hasattr(self, 'w'):
            self.w = 7

    def get_ssm_matrices(self, update_rate=0.1):
        '''
        For the linear stochastic state-space model 
            x_{t+1} = Ax_{t} + Bu_t + w_t;   w_t ~ N(0, W),
        this function specifies the matrices A, B and W

        A = [I_N    \Delta I_N   0
             0_N    a*I_N        0
             0      0            1]

        W = [0_N    0_N   0
             0_N    w*I_N        0
             0      0            0]

        B = [0_N
            1000\Delta I_N
            0]

        Parameters
        ----------
        update_rate : float, optional
            Time between iterations of the discrete-time model. Default is 0.1 sec.

        Returns
        -------
        tuple of 3 np.mat matrices
            A, B and W as specified in the mathematical model above
        '''
        if not (update_rate is None):
            a_resamp, w_resamp = resample_scalar_ssm(self.vel_decay, self.w, Delta_old=self.Delta, Delta_new=update_rate)
            Delta = update_rate
        else:
            a_resamp = self.vel_decay
            w_resamp = self.w
            Delta = self.Delta

        ndim = len(np.nonzero(self.state_order == 1)[0])
        A = _gen_A(1, Delta, 0, a_resamp, 1, ndim=ndim)
        W = _gen_A(0, 0, 0, w_resamp, 0, ndim=ndim)        

        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(ndim))
        B = np.vstack([0*I, Delta*1000 * I, np.zeros([1, ndim])])

        # account for offset state
        has_offset = self.states[-1].order == -1
        if not has_offset:
            A = A[:-1, :-1]
            W = W[:-1, :-1]
            B = B[:-1, :]

        return A, B, W

    def __eq__(self, other):
        states_equal = super(LinearVelocityStateSpace, self).__eq__(other)
        A1, B1, W1 = self.get_ssm_matrices()
        A2, B2, W2 = other.get_ssm_matrices()
        return states_equal and np.array_equal(A1, A2) and np.array_equal(B1, B2) and np.array_equal(W1, W2)

offset_state = State('offset', stochastic=False, drives_obs=True, order=-1)

class StateSpaceNLinkPlanarChain(LinearVelocityStateSpace):
    '''
    State-space model for an N-link kinematic chain
    '''
    def __init__(self, n_links=2, **kwargs):
        self.n_links = n_links
        pos_states = []
        vel_states = []

        for k in range(n_links):
            pos_state_k = State('theta_%d' % k, stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0)
            vel_state_k = State('omega_%d' % k, stochastic=True, drives_obs=True, order=1)
            pos_states.append(pos_state_k)
            vel_states.append(vel_state_k)

        states = pos_states + vel_states + [offset_state]
        super(StateSpaceNLinkPlanarChain, self).__init__(states, **kwargs)

    def __setstate__(self, state):
        self.__dict__ = state
        if not hasattr(self, 'Delta'):
            self.Delta = 0.1

        if not hasattr(self, 'vel_decay'):
            self.vel_decay = 0.8

        if not hasattr(self, 'w'):
            self.w = 0.01

class StateSpaceEndptVel2D(StateSpace):
    '''
    StateSpace with 2D velocity in the X-Z plane
    '''
    def __init__(self):
        super(StateSpaceEndptVel2D, self).__init__(
            State('hand_px', stochastic=False, drives_obs=False, min_val=-25., max_val=25., order=0),
            State('hand_py', stochastic=False, drives_obs=False, order=0),
            State('hand_pz', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
            State('hand_vx', stochastic=True,  drives_obs=True, order=1),
            State('hand_vy', stochastic=False, drives_obs=False, order=1),
            State('hand_vz', stochastic=True,  drives_obs=True, order=1),
            offset_state
        )

    def get_ssm_matrices(self, update_rate=0.1):
        '''
        For the linear stochastic state-space model 
            x_{t+1} = Ax_{t} + Bu_t + w_t;   w_t ~ N(0, W),
        this function specifies the matrices A, B and W

        Parameters
        ----------
        update_rate : float, optional
            Time between iterations of the discrete-time model. Default is 0.1 sec.

        Returns
        -------
        tuple of 3 np.mat matrices
            A, B and W as specified in the mathematical model above
        '''
        # State-space model set from expert data
        A, W = linear_kinarm_kf(update_rate=update_rate)

        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(3))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1,3])])
        return A, B, W

if __name__ == '__main__':
    a_10hz = 0.8
    w_10hz = 0.0007

    Delta_old = 0.1
    Delta_new = 1./60
    a_60hz, w_60hz = resample_scalar_ssm(a_10hz, w_10hz, Delta_old=Delta_old, Delta_new=Delta_new)