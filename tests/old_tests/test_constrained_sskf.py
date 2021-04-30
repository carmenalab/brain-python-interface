#!/usr/bin/python
"""
Test case for SDVKF implementation in python
"""
from scipy.io import loadmat
import riglib.bmi.kfdecoder
import pickle
import numpy as np

decoder_file = '/storage/decoders/cart20130902_01_AI09021528.pkl'
decoder = pickle.load(open(decoder_file))
C = decoder.kf.C
A = decoder.kf.A
W = decoder.kf.W
Q_hat = decoder.kf.Q
is_stochastic = decoder.kf.is_stochastic
drives_neurons = np.array([False, False, True, True, True])

# TODO extension: separate out the equality constraints and the orthogonality constraints
Q = riglib.bmi.kfdecoder.project_Q(C[:,2:4], Q_hat)

# create a riglib.bmi.KalmanFilter object and compute the steady-state Kalman filter
from riglib.bmi.kfdecoder import KalmanFilter
kf = KalmanFilter(A, W, C, Q_hat)
[F, K] = kf.get_sskf()

# In one of the other arguments to the decoder updater?
C_new = C.copy()
C_new[:, ~drives_neurons] = 0
kf_new = KalmanFilter(A, W, C_new, Q)
F_new, K_new = kf_new.get_sskf()
print("\n\n\n")
print("T=")
print(F_new[0:2,0:2])
print("S=")
print(F_new[0:2,2:4])
print("N=")
print(F_new[2:4,2:4])
print("M=")
print(F_new[2:4,0:2])
