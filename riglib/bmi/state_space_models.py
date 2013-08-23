#!/usr/bin/python
"""
(Soon to be) a collection of state-space models for decoders
"""
import numpy as np

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
    print mu
    print sigma
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

def linear_kinarm_kf(update_rate=1./10):
    Delta_KINARM = 1./10
    loop_update_ratio = update_rate/Delta_KINARM
    a_resampled, w_resampled = resample_scalar_ssm(0.8, 700, Delta_old=Delta_KINARM, Delta_new=update_rate)
    A = _gen_A(1, update_rate, 0, a_resampled, 1, ndim=3)
    W = _gen_A(0, 0, 0, w_resampled, 0, ndim=3)
    return A, W
    

if __name__ == '__main__':
    a_10hz = 0.8
    w_10hz = 0.0007

    Delta_old = 0.1
    Delta_new = 1./60
    a_60hz, w_60hz = resample_scalar_ssm(a_10hz, w_10hz, Delta_old=Delta_old, Delta_new=Delta_new)
