"""Calculations of performance metrics for 2-D reaching tasks"""
import numpy as np
from scipy.stats import circmean
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict

import os
import tables
from riglib.bmi import robot_arms, train


min_per_sec = 1./60
seconds_per_min = 60
sec_per_min = 60

pi = np.pi
plot_dir = '/storage/plots'

def _count_switches(vec):
    """ vec is an array of binary variables (0,1). The number of switches
    between 1's and 0's is counted
    """
    return len(np.nonzero(edge_detect(vec, 'pos'))[0]) + len(np.nonzero(edge_detect(vec, 'neg'))[0])

def edge_detect(vec, edge_type='pos'):
    """ Edge detector for a 1D array

    Example:

    vec = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...]
                       ^           ^
                       ^           ^
                      pos         neg
                      edge        edge

    vec         : 1D array
    edge_type   : {'pos', 'neg'}
    """
    if np.ndim(vec) > 1:
        vec = vec.reshape(-1)
    T = len(vec)
    edges = np.zeros(T)
    for t in range(1,T):
        if edge_type == 'pos':
            if vec[t] and not vec[t-1]:
                edges[t] = 1
        elif edge_type == 'neg':
            if vec[t-1] and not vec[t]:
                edges[t] = 1
    return edges

def get_goal_error_basis(cursor, target):
    '''
    Define a coordinate system where one axis is along the straight-line
    from the current position to the target, and the other is orthogonal.
    Only for 2-D tasks with movement in the X-Z plane (no "depth")

    Parameters
    ----------
    cursor : np.ndarray of shape (3,)
    target : np.ndarray of shape (3,)

    Returns
    -------
    vec_to_targ : np.ndarray of shape (2,)
    orth_vec_to_targ : np.ndarray of shape (2,)
    '''
    # 90 degree rotation matrix
    R = np.array([[0, -1],
                  [1, 0]])

    vec_to_targ = target - cursor
    vec_to_targ = vec_to_targ[[0,2]]
    vec_to_targ /= np.linalg.norm(vec_to_targ)

    orth_vec_to_targ = np.dot(R, vec_to_targ)
    orth_vec_to_targ /= np.linalg.norm(orth_vec_to_targ)
    return vec_to_targ, orth_vec_to_targ

def get_task_axis_error_measures(traj, origin=np.array([0., 0.]),
    target=np.array([10., 0.]), sl=slice(None,None,1), eff_target_size=1.7-0.48,
    return_type='dict', data_rate=60., rt_from_exit_origin=False):
    '''
    calculate ME, MV, MDC, ODC, path length, and normalized path length
    for a trajectory
    '''

    if not 2 in traj.shape:
        assert "Wrong shape for trajectory!"
    elif traj.shape[1] == 2:
        traj = traj.T

    # import pdb; pdb.set_trace()

    if not np.all(origin == 0):
        traj = (traj.T - origin).T
        target -= origin
        origin = np.zeros(2)

    angle = -np.arctan2(target[1], target[0])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    traj = np.dot(R, traj)
    ME = np.mean(np.abs(traj[1, sl]))
    MV = np.std(np.abs(traj[1, sl]))
    ODC = _count_switches( 0.5*(np.sign(np.diff(traj[0, sl])) + 1) )
    MDC = _count_switches( 0.5*(np.sign(np.diff(traj[1, sl])) + 1) )
    PL = np.sum(map(np.linalg.norm, np.diff(traj, axis=1).T))
    NPL = PL / np.linalg.norm(target - origin)

    # Determine when the cursor exits the center for the first time, i.e. ||pos - center|| > (target_radius - cursor_radius)
    dist_from_origin = np.array(map(lambda x: np.linalg.norm(x - origin), traj.T))
    try:
        if rt_from_exit_origin:
            exit_origin_ind = np.nonzero(dist_from_origin > eff_target_size)[0][0]
        else:
            exit_origin_ind = 0
        reach_time = (traj.shape[1] - exit_origin_ind)*1./data_rate#sl.step
    except:
        reach_time = np.nan

    if return_type == 'dict':
        perf = OrderedDict()

    elif return_type == 'recarray':
        data = ['ME', 'MV', 'PL', 'NPL', 'MDC', 'ODC', 'reach_time']
        dtype = np.dtype([(x, 'f8') for x in data])
        perf = np.zeros(1, dtype=dtype)
    else:
        raise ValueError("Unrecognized return type: %s" % return_type)
    perf['PL'] = PL
    perf['ME'] = ME
    perf['MV'] = MV
    perf['NPL'] = NPL
    perf['MDC'] = MDC
    perf['ODC'] = ODC
    perf['reach_time'] = reach_time
    return perf
