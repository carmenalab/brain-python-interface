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

def resample_ssm(A, W, Delta_old=0.1, Delta_new=0.005, include_offset=True):
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
    """utility function for generating block-diagonal matrices
    used by the KF
    """
    A = np.zeros([2*ndim+1, 2*ndim+1])
    A_lower_dim = np.array([[t, s], [m, n]])
    A[0:2*ndim, 0:2*ndim] = np.kron(A_lower_dim, np.eye(ndim))
    A[-1,-1] = off
    return np.mat(A)

def linear_kinarm_kf(update_rate=1./10, units_mult=0.01):
    Delta_KINARM = 1./10
    loop_update_ratio = update_rate/Delta_KINARM
    w_in_meters = 0.0007
    w_units_resc = w_in_meters / (units_mult ** 2)
    a_resampled, w_resampled = resample_scalar_ssm(0.8, w_units_resc, Delta_old=Delta_KINARM, Delta_new=update_rate)
    A = _gen_A(1, update_rate, 0, a_resampled, 1, ndim=3)
    W = _gen_A(0, 0, 0, w_resampled, 0, ndim=3)
    return A, W
    

class State(object):
    def __init__(self, name, stochastic=False, drives_obs=False, min_val=np.nan, max_val=np.nan, order=-1):
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
        if not isinstance(other, State):
            return False
        else:
            return np.all([self.__dict__[x] == other.__dict__[x] or (np.isnan(self.__dict__[x]) and np.isnan(other.__dict__[x])) for x in self.__dict__])
            # return self.__dict__

class StateSpace(object):
    def __init__(self, *states):
        self.states = list(states)

    def __repr__(self):
        return 'State space: ' + str(self.state_names)

    @property
    def is_stochastic(self):
        return np.array([x.stochastic for x in self.states])

    @property
    def drives_obs(self):
        return np.array([x.drives_obs for x in self.states])

    @property
    def state_names(self):
        return [x.name for x in self.states]

    @property
    def bounding_box(self):
        min_bounds = np.array(filter(lambda x: x is not np.nan, [x.min_val for x in self.states]))
        max_bounds = np.array(filter(lambda x: x is not np.nan, [x.max_val for x in self.states]))
        return (min_bounds, max_bounds)

    @property
    def states_to_bound(self):
        return [x.name for x in filter(lambda x: x.min_val is not np.nan, self.states)]

    @property
    def n_states(self):
        return len(self.states)

    @property
    def train_inds(self):
        return filter(lambda k: self.states[k].stochastic, range(self.n_states))

    @property
    def drives_obs_inds(self):
        return filter(lambda k: self.states[k].drives_obs, range(self.n_states))

    @property 
    def state_order(self):
        return np.array([x.order for x in self.states])

    def get_ssm_matrices(self):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, StateSpace):
            return False
        else:
            return self.states == other.states

offset_state = State('offset', stochastic=False, drives_obs=True, order=-1)

class StateSpaceEndptVel(StateSpace):
    def __init__(self):
        super(StateSpaceEndptVel, self).__init__(
            State('hand_px', stochastic=False, drives_obs=False, min_val=-25., max_val=25., order=0),
            State('hand_py', stochastic=False, drives_obs=False, order=0),
            State('hand_pz', stochastic=False, drives_obs=False, min_val=-14., max_val=14., order=0),
            State('hand_vx', stochastic=True,  drives_obs=True, order=1),
            State('hand_vy', stochastic=False, drives_obs=False, order=1),
            State('hand_vz', stochastic=True,  drives_obs=True, order=1),
            offset_state
        )

    def get_ssm_matrices(self, update_rate=0.1):
        # State-space model set from expert data
        A, W = state_space_models.linear_kinarm_kf(update_rate=update_rate)

        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(3))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1,3])])
        return A, B, W

class StateSpaceExoArm(StateSpace):
    '''
    State space representing the kinematics of the exoskeleton
        1) shoulder flexion extension
        2) shoulder abduction/adduction
        3) elbow rotation
        4) elbow flexion/extension
        5) pronation/supination
    '''
    def __init__(self):
        super(StateSpaceExoArm, self).__init__(
                # position states
                State('sh_pflex', stochastic=False, drives_obs=False, order=0),
                State('sh_pabd', stochastic=False, drives_obs=False, order=0), 
                State('sh_prot', stochastic=False, drives_obs=False, order=0), 
                State('el_pflex', stochastic=False, drives_obs=False, order=0), 
                State('el_psup', stochastic=False, drives_obs=False, order=0), 
                # velocity states
                State('sh_vflex', stochastic=True, drives_obs=True, order=1), 
                State('sh_vabd', stochastic=True, drives_obs=True, order=1), 
                State('sh_vrot', stochastic=True, drives_obs=True, order=1), 
                State('el_vflex', stochastic=True, drives_obs=True, order=1), 
                State('el_vsup', stochastic=True, drives_obs=True, order=1), 
                # offset
                offset_state,
        )

    def get_ssm_matrices(self, update_rate=0.1):
        raise NotImplementedError("Still need to determine A for the full joint space. Need 3D reaching data from real primate")

class StateSpaceExoArm2D(StateSpaceExoArm):
    '''
    Exo arm, but limited to the 2D x-z plane by allowing only 
    should abduction/adduction and elbow flexion/extension
    '''
    def __init__(self):
        super(StateSpaceExoArm, self).__init__(
                # position states
                State('sh_pflex', stochastic=False, drives_obs=False, min_val=0, max_val=0, order=0),
                State('sh_pabd', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('sh_prot', stochastic=False, drives_obs=False, min_val=0, max_val=0, order=0),
                State('el_pflex', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('el_psup', stochastic=False, drives_obs=False, min_val=0, max_val=0, order=0),
                # velocity states
                State('sh_vflex', stochastic=False, drives_obs=False, order=1),
                State('sh_vabd', stochastic=True, drives_obs=True, order=1),
                State('sh_vrot', stochastic=False, drives_obs=False, order=1),
                State('el_vflex', stochastic=True, drives_obs=True, order=1),
                State('el_vsup', stochastic=False, drives_obs=False, order=1),
                # offset
                offset_state,
        )

    def get_ssm_matrices(self, update_rate=0.1):
        '''
        State space model from expert data
        '''
        Delta_KINARM = 1./10
        w = 0.01 #0.0007
        #w = 0.3 # TODO come up with this value more systematically!
        w_units_resc = w / 1 # velocity will always be in radians/sec
        a_resampled, w_resampled = state_space_models.resample_scalar_ssm(0.8, w_units_resc, Delta_old=Delta_KINARM, Delta_new=update_rate)

        # TODO get the number of dimensions from the arm configuration (i.e. a method to return the order of each state
        ndim = 5 # NOTE: This is the number of 1st order states, not the dimension of the state vector
        A = state_space_models._gen_A(1, update_rate, 0, a_resampled, 1, ndim=ndim)
        W = state_space_models._gen_A(0, 0, 0, w_resampled, 0, ndim=ndim)
        
        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(ndim))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1, ndim])])
        return A, B, W

class StateSpaceFourLinkTentacle2D(StateSpace):
    def __init__(self):
        super(StateSpaceFourLinkTentacle2D, self).__init__(
                # position states
                State('sh_pabd', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('el_pflex', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('wr_pflex', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                State('fi_pflex', stochastic=False, drives_obs=False, min_val=-pi, max_val=0, order=0),
                # velocity states
                State('sh_vabd', stochastic=True, drives_obs=True, order=1),
                State('el_vflex', stochastic=True, drives_obs=True, order=1),
                State('wr_vflex', stochastic=True, drives_obs=True, order=1),
                State('fi_vflex', stochastic=True, drives_obs=True, order=1),
                # offset
                offset_state,
        )

    def get_ssm_matrices(self, update_rate=0.1):
        '''
        State space model from expert data
        '''
        Delta_KINARM = 1./10
        w = 0.01 #0.0007
        #w = 0.3 # TODO come up with this value more systematically!
        w_units_resc = w / 1 # velocity will always be in radians/sec
        a_resampled, w_resampled = state_space_models.resample_scalar_ssm(0.8, w_units_resc, Delta_old=Delta_KINARM, Delta_new=update_rate)

        # TODO get the number of dimensions from the arm configuration (i.e. a method to return the order of each state
        ndim = 4 # NOTE: This is the number of 1st order states, not the dimension of the state vector
        A = state_space_models._gen_A(1, update_rate, 0, a_resampled, 1, ndim=ndim)
        W = state_space_models._gen_A(0, 0, 0, w_resampled, 0, ndim=ndim)
        
        # Control input matrix for SSM for control inputs
        I = np.mat(np.eye(ndim))
        B = np.vstack([0*I, update_rate*1000 * I, np.zeros([1, ndim])])
        return A, B, W        


if __name__ == '__main__':
    a_10hz = 0.8
    w_10hz = 0.0007

    Delta_old = 0.1
    Delta_new = 1./60
    a_60hz, w_60hz = resample_scalar_ssm(a_10hz, w_10hz, Delta_old=Delta_old, Delta_new=Delta_new)
