'''
Feedback controllers for assist/simulation purposes
'''
import numpy as np

class CenterOutCursorGoal():
    def __init__(self, angular_noise_var=0):
        self.interactive = False
        self.angular_noise_var = angular_noise_var

        #noise_mdl_fh = open('/Users/surajgowda/bmi/lib/ops_cursor/seba_int_vel_noise_model.dat', 'rb')
        #self.noise_model = pickle.load(noise_mdl_fh)

    def get(self, cur_target, cur_pos, keys_pressed=None, gain=0.15):
        dir_to_targ = cur_target - cur_pos

        if self.angular_noise_var > 0:
            #angular_noise_rad = self.noise_model.sample()[0,0]
            angular_noise_rad = np.random.normal(0, self.angular_noise_var)
            while abs(angular_noise_rad) > np.pi:
                #angular_noise_rad = self.noise_model.sample()[0,0]
                anglular_noise_rad = np.random.normal(0, self.angular_noise_var)
        else:
            angular_noise_rad = 0
        angular_noise = np.array([np.cos(angular_noise_rad), np.sin(angular_noise_rad)])     
        return gain*( dir_to_targ/np.linalg.norm(dir_to_targ) + angular_noise )
