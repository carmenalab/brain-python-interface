'''
Feedback controllers for assist/simulation purposes
'''
import numpy as np

class CenterOutCursorGoal():
    def __init__(self, angular_noise_var=0):
        self.interactive = False
        self.angular_noise_var = angular_noise_var

    def get(self, cur_target, cur_pos, keys_pressed=None, gain=0.15):
        dir_to_targ = cur_target - cur_pos

        if self.angular_noise_var > 0:
            angular_noise_rad = np.random.normal(0, self.angular_noise_var)
            while abs(angular_noise_rad) > np.pi:
                anglular_noise_rad = np.random.normal(0, self.angular_noise_var)
        else:
            angular_noise_rad = 0
        angular_noise = np.array([np.cos(angular_noise_rad), np.sin(angular_noise_rad)])     
        return gain*( dir_to_targ/np.linalg.norm(dir_to_targ) + angular_noise )

def dlqr(A, B, Q, R, Q_f=None, T=np.inf, max_iter=1000, eps=1e-5, dtype=np.mat):
    if Q_f == None: 
        Q_f = Q

    if T < np.inf: # Finite horizon
        K = [None]*T
        P = Q_f
        for t in range(0,T-1)[::-1]:
            K[t] = (R + B.T*P*B).I * B.T*P*A
            P = Q + A.T*P*A -A.T*P*B*K[t]
        return dtype(K)
    else: # Infinite horizon
        P = Q_f
        K = np.inf
        for t in range(max_iter):
            K_old = K
            K = (R + B.T*P*B).I * B.T*P*A
            P = Q + A.T*P*A -A.T*P*B*K 
            if np.linalg.norm(K - K_old) < eps:
                break
        return dtype(K)

