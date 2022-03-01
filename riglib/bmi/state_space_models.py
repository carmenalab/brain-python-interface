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


class State(object):
    '''
    A 1D component of a state-space, e.g., vertical velocity
    '''
    def __init__(self, name, stochastic=False, drives_obs=False, min_val=np.nan, max_val=np.nan, order=-1, aux=False):
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
            e.g., integrated position is order -1, position states are order 0, velocity states are order 1, acceleration is order 2, etc.
            Constant states should be order NaN

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
        self.aux = aux
        self._eq_comp_excl = []

    def __repr__(self):
        return str(self.name) 

    def __eq__(self, other):
        '''
        State instances are equal if all their attributes (name, stochastic, etc.) are equal
        '''
        if not isinstance(other, State):
            return False
        else:
            for x in self.__dict__:
                if x in other.__dict__:
                    if not (x == '_eq_comp_excl') and not (x in self._eq_comp_excl) and not (x in other._eq_comp_excl):
                        if not (self.__dict__[x] == other.__dict__[x] or (np.isnan(self.__dict__[x]) and np.isnan(other.__dict__[x]))):
                            print(self, other, x)
                            import pdb; pdb.set_trace()
                            return False
            return True

    def __setstate__(self, data):
        self.__dict__ = data
        if '_eq_comp_excl' not in self.__dict__:
            self._eq_comp_excl = []


class StateSpace(object):
    '''
    A collection of multiple 'State' instances forms a StateSpace
    '''
    def __init__(self, *states, **kwargs):
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
        if 'statelist' in kwargs:
            self.states = kwargs['statelist']
        else:
            self.states = list(states)

    def __len__(self):
        return len(self.states)
        
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
    def is_aux_state(self):
        return np.array([x.aux for x in self.states])

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
        min_bounds = np.array([x for x in [x.min_val for x in self.states] if x is not np.nan])
        max_bounds = np.array([x for x in [x.max_val for x in self.states] if x is not np.nan])
        return (min_bounds, max_bounds)

    @property
    def states_to_bound(self):
        '''
        A list of the names of all the states which have limits on the values they can take.
        '''
        return [x.name for x in [x for x in self.states if x.min_val is not np.nan]]

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
        return [k for k in range(self.n_states) if self.states[k].stochastic]

    @property
    def drives_obs_inds(self):
        '''
        A list of the indices of the states which are related to observations when 
        used as a hidden state-space. Used when seeding Decoders
        '''
        return [k for k in range(self.n_states) if self.states[k].drives_obs]

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
        has_offset = self.states[-1] == offset_state
        # has_offset = np.isnan(self.states[-1].order)
        if not has_offset:
            A = A[:-1, :-1]
            W = W[:-1, :-1]
            B = B[:-1, :]

        return A, B, W

    def __eq__(self, other):
        states_equal = super(LinearVelocityStateSpace, self).__eq__(other)
        A1, B1, W1 = self.get_ssm_matrices()
        A2, B2, W2 = other.get_ssm_matrices()
        # import pdb; pdb.set_trace()
        return states_equal and np.array_equal(A1, A2) and np.array_equal(B1, B2) and np.array_equal(W1, W2)


#######################################################################
##### Specific StateSpace types for particular experiments/plants #####
#######################################################################
# These class declarations may not actually be best placed in this class, 
# but moving them now would cause problems with unpickling older decoder 
# objects, which are saved with these state space model types. So put new 
# state-space models elsewhere!
offset_state = State('offset', stochastic=False, drives_obs=True, order=np.nan)
offset_state._eq_comp_excl.append('order')

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

class StateSpaceEndptVel2D(LinearVelocityStateSpace):
    '''
    StateSpace with 2D velocity in the X-Z plane
    '''
    def __init__(self, **kwargs):
        states = [
            State('hand_px', stochastic=False, drives_obs=False, min_val=-25., max_val=25., order=0),
            State('hand_py', stochastic=False, drives_obs=False, order=0),
            State('hand_pz', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
            State('hand_vx', stochastic=True,  drives_obs=True, order=1),
            State('hand_vy', stochastic=False, drives_obs=False, order=1),
            State('hand_vz', stochastic=True,  drives_obs=True, order=1),
            offset_state]
        super(StateSpaceEndptVel2D, self).__init__(states, **kwargs)

    def __setstate__(self, state):
        self.__dict__ = state
        if not hasattr(self, 'Delta'):
            self.Delta = 0.1

        if not hasattr(self, 'vel_decay'):
            self.vel_decay = 0.8

        if not hasattr(self, 'w'):
            self.w = 7

class StateSpaceEndptVel3D(LinearVelocityStateSpace):
    def __init__(self, **kwargs):
        states = [
            State('hand_px', stochastic=False, drives_obs=False, min_val=-25., max_val=25., order=0),
            State('hand_py', stochastic=False, drives_obs=False, order=0),
            State('hand_pz', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
            State('hand_vx', stochastic=True,  drives_obs=True, order=1),
            State('hand_vy', stochastic=True,  drives_obs=True, order=1),
            State('hand_vz', stochastic=True,  drives_obs=True, order=1),
            offset_state]
        super(StateSpaceEndptVel3D, self).__init__(states, **kwargs)  
    def __setstate__(self, state):
        self.__dict__ = state
        if not hasattr(self, 'Delta'):
            self.Delta = 0.1

        if not hasattr(self, 'vel_decay'):
            self.vel_decay = 0.8

        if not hasattr(self, 'w'):
            self.w = 7  

class StateSpaceEndptPos1D(StateSpace):
    ''' StateSpace for 1D pos control (e.g. RatBMI)'''
    def __init__(self, **kwargs):
        states = State('cursor_p', stochastic=False, drives_obs=True, min_val=-10e6, max_val=10e6, order=0)
            
        super(StateSpaceEndptPos1D, self).__init__(states, **kwargs)

    def __setstate__(self, state):
        self.__dict__ = state
        if not hasattr(self, 'Delta'):
            self.Delta = 0.1

        if not hasattr(self, 'vel_decay'):
            self.vel_decay = 0.8

        if not hasattr(self, 'w'):
            self.w = 7

############################
##### Helper functions #####
############################
def resample_ssm(A, W, Delta_old=0.1, Delta_new=0.005, include_offset=True):
    '''
    Change the effective sampling rate of a linear random-walk discrete-time state-space model
    That is, state-space models of the form 

    x_{t+1} = Ax_t + w_t, w_t ~ N(0, W)

    Parameters
    ----------
    A : np.mat of shape (K, K)
        State transition model
    W : np.mat of shape (K, K)
        Noise covariance estimate
    Delta_old : float, optional, default=0.1
        Old sampling rate
    Delta_new : float, optional, default=0.005
        New sampling rate
    include_offset : bool, optional, default=True
        Indicates whether the state-space matrices 

    Returns
    -------
    A_new, W_new
        New state-space model parameters at the new sampling rate.

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
    Similar to resample_ssm, but for a scalar (1-d) state-space model, 
    where the problem can be solved without complicated matrix roots

    Parameters
    ----------
    a : float
        State transition model
    w : float 
        Noise variance estimate
    Delta_old : float, optional, default=0.1
        Old sampling rate
    Delta_new : float, optional, default=0.005
        New sampling rate

    Returns
    -------
    a_new, w_new
        New state-space model parameters at the new sampling rate.

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
    t : float 
        See matrix equation above
    s : float 
        See matrix equation above
    m : float 
        See matrix equation above
    n : float 
        See matrix equation above
    off : float 
        See matrix equation above
    ndim : int, optional, default = 3 
        Number of states in each block, e.g. 3-states for (x,y,z) position

    Returns
    -------
    np.mat of shape (N, N); N = 2*ndim + 1
    """
    A = np.zeros([2*ndim+1, 2*ndim+1])
    A_lower_dim = np.array([[t, s], [m, n]])
    A[0:2*ndim, 0:2*ndim] = np.kron(A_lower_dim, np.eye(ndim))
    A[-1,-1] = off
    return np.mat(A)


if __name__ == '__main__':
    a_10hz = 0.8
    w_10hz = 0.0007

    Delta_old = 0.1
    Delta_new = 1./60
    a_60hz, w_60hz = resample_scalar_ssm(a_10hz, w_10hz, Delta_old=Delta_old, Delta_new=Delta_new)