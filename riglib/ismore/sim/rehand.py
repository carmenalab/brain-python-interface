'''ReHand simulation code.

Models ReHand as a plant whose (angular) velocities can be changed
instantaneously, and whose (angular) positions are updated based on
the integral of velocity. The ReHand runs in a separate process and
automatically "steps" itself as time passes.
'''

import time
import multiprocessing as mp
from sim_utils import shared_np_mat
import numpy as np
from math import sin, cos

pi = np.pi


class ReHand(mp.Process):
    '''Order of joints is always: thumb, index, fing3, prono. All units in rad/s.'''
    def __init__(self, tstep, t_print=15):
        super(ReHand, self).__init__()

        # ReHand moves itself every tstep
        self.tstep = tstep  

        dtype = np.dtype('float64')
        self.pos = shared_np_mat(dtype, (4, 1))  # pos
        self.vel = shared_np_mat(dtype, (4, 1))  # vel

        self.active = mp.Value('b', 0)
        self.lock = mp.Lock()

        self.t_print = t_print

    def run(self):
        print 'starting ReHand'
        self.active.value = 1
        t_start = time.time()
        t_simulated = 0  # how much time has been simulated

        # prints its clock every t_print secs
        next_t_print = self.t_print
        
        while self.active.value == 1:
            # how much real time has elapsed
            t_elapsed = time.time() - t_start

            if t_elapsed - t_simulated >= self.tstep:
                self.step()
                t_simulated += self.tstep
            else:
                # this might not be necessary, maybe we could just "pass" here
                time.sleep(self.tstep/2)

            if t_elapsed > next_t_print:
                print 'ReHand at t=%3.3f s' % t_elapsed
                next_t_print += self.t_print

    def step(self):
        with self.lock:
            self.pos[:, :] = self.pos.copy() + self.tstep*self.vel.copy()

    def stop(self):
        self.active.value = 0

    def get_state(self):
        state = dict()
        with self.lock:
            state['pos'] = self.pos.copy()
            state['vel'] = self.vel.copy()

        return state

    def set_vel(self, vel):
        with self.lock:
            self.vel[:, :] = np.mat(vel).reshape((4, 1))

    # IMPORTANT: only use to set starting position
    def _set_pos(self, pos):
        with self.lock:
            self.pos[:, :] = np.mat(pos).reshape((4, 1))
