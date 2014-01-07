'''
Feedback controllers for assist/simulation purposes
'''
import numpy as np

class CenterOutCursorGoal(object):
    def __init__(self, angular_noise_var=0, gain=0.15):
        self.interactive = False
        self.angular_noise_var = angular_noise_var
        self.gain = gain

    def get(self, cur_target, cur_pos, keys_pressed=None):
        # Make sure y-dimension is 0
        assert cur_pos[1] == 0
        assert cur_target[1] == 0

        dir_to_targ = cur_target - cur_pos

        if self.angular_noise_var > 0:
            angular_noise_rad = np.random.normal(0, self.angular_noise_var)
            while abs(angular_noise_rad) > np.pi:
                angular_noise_rad = np.random.normal(0, self.angular_noise_var)
        else:
            angular_noise_rad = 0
        #angular_noise = np.array([np.cos(angular_noise_rad), np.sin(angular_noise_rad)])     
        angle = np.arctan2(dir_to_targ[2], dir_to_targ[0])
        sum_angle = angle + angular_noise_rad
        return self.gain*np.array([np.cos(sum_angle), np.sin(sum_angle)])
        #return gain*( dir_to_targ/np.linalg.norm(dir_to_targ) + angular_noise )

class CenterOutCursorGoalJointSpace2D(CenterOutCursorGoal):
    def __init__(self, link_lengths, shoulder_anchor, *args, **kwargs):
        self.link_lengths = link_lengths
        self.shoulder_anchor = shoulder_anchor
        super(CenterOutCursorGoalJointSpace2D, self).__init__(*args, **kwargs)


    def get(self, cur_target, cur_pos, keys_pressed=None):
        '''
        cur_target and cur_pos should be specified in workspace coordinates
        '''
        vx, vz = super(CenterOutCursorGoalJointSpace2D, self).get(cur_target, cur_pos, keys_pressed)
        vy = 0

        px, py, pz = cur_pos

        pos = np.array([px, py, pz]) - self.shoulder_anchor
        vel = np.array([vx, vy, vz])

        # Convert to joint velocities
        from riglib.stereo_opengl import ik
        joint_pos, joint_vel = ik.inv_kin_2D(pos, self.link_lengths[0], self.link_lengths[1], vel)
        return joint_vel[0]['sh_vabd'], joint_vel[0]['el_vflex']



def dlqr(A, B, Q, R, Q_f=None, T=np.inf, max_iter=1000, eps=1e-10, dtype=np.mat):
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

