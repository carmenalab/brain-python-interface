'''ArmAssist simulation code.

Main parts were originally written in MATLAB by Je Hyung Jung at Tecnalia
(jehyung.jung@tecnalia.com). Code has been translated into Python and adapted
so that ArmAssist and ArmAssistPIController run in separate processes and
automatically "step" themselves as time passes.
'''

import time
import multiprocessing as mp
from sim_utils import shared_np_mat
import numpy as np
from math import sin, cos

pi = np.pi
cm_to_m = 0.01
m_to_cm = 100.


# mechanical constants of robot - don't change values (copied directly from Je's code)
m = 3.05 # m - Robot mass (kg)
Iz = 0.067 # Iz - Robot moment inertia (kgm2) ! this is not actual value
R = 0.0254 # R - Wheel radius (m)
L1 = 0.130 # L1 - Distance from the center mouse to front wheel.
L2 = 0.153 # L2 - Distance from the center mouse to rear right (left) wheel.
n = 19 # n - Gear ratio
J0 = 0.003 # combined inertia of the motor, gear train and wheel referred to the motor shaft ! this is not acutal value
b0 = 0 # viscous-friction coefficient of the motor, gear and wheel combination

# matrices used in the model - don't change them. If you need to change, please let me know.
H = np.mat([[1./m, 0,    0], 
            [0,    1./m, 0],
            [0,    0,    1./Iz]])
B = np.mat([[-1,  cos(pi/4),        cos(pi/4)], 
            [0,  -sin(pi/4),        sin(pi/4)],
            [L1, L2*cos(pi*12/180), L2*cos(pi*12/180)]])
G = np.identity(3) + (H * B * B.T * n**2 * J0 / R**2)


# used to avoid repeated computation of these values
G_I = G.I
constant1 = H * B * B.T * b0 * n**2 / R**2
constant2 = H * B * n / R


def dyn_eq_body_frame(torque, bf, f_ex):
    '''Dynamic model of ArmAssist in the body frame.'''
    bf_ = np.array(bf).reshape(3)
    tmp = np.mat([bf_[2]*bf_[1], -bf_[2]*bf_[0], 0]).T
    bf_dot = G_I * (tmp - constant1*bf + constant2*torque) + f_ex  # equation of motion
    return bf_dot

def kine_eq_robot(bf, a):
    '''Kinematics of ArmAssist in world frame.'''
    wf_dot = np.mat([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]]) * bf
    return wf_dot

def rungekutta_body_frame(tor, bf, f_ex, tstep):
    '''Rungekutta integration of dynamics in the body frame...'''
    X = bf
    torque = tor

    x_dot1 = dyn_eq_body_frame(torque, X, f_ex)
    x = X + x_dot1*tstep/2
    x_dot2 = dyn_eq_body_frame(torque, x, f_ex)
    x = X + x_dot2*tstep/2
    x_dot3 = dyn_eq_body_frame(torque, x, f_ex)
    x = X + x_dot3*tstep
    x_dot4 = dyn_eq_body_frame(torque, x, f_ex)
    X = X + tstep*((x_dot1+x_dot4)/2 + x_dot2 + x_dot3)/3

    q = X
    q_dot = x_dot1

    return q, q_dot

def rungekutta_world_frame(wf, bf, tstep):
    '''Rungekutta integration of kinematics in the world frame.'''
    X = wf

    x_dot1 = kine_eq_robot(bf, X[2])
    x = X + x_dot1*tstep/2
    x_dot2 = kine_eq_robot(bf, x[2])
    x = X + x_dot2*tstep/2
    x_dot3 = kine_eq_robot(bf, x[2])
    x = X + x_dot3*tstep
    x_dot4 = kine_eq_robot(bf, x[2])
    X = X + tstep*((x_dot1+x_dot4)/2 + x_dot2 + x_dot3)/3

    q = X
    q_dot = x_dot1

    return q, q_dot


class ArmAssist(mp.Process):
    '''
    Simulated ArmAssist object that runs in its own process and automatically
    updates its position and velocity as time passes, based upon whatever its
    current torque value is (which is updated using an ArmAssistPIController).
    '''
    def __init__(self, aa_tstep, aa_pic_tstep, KP, TI, t_print=15):
        super(ArmAssist, self).__init__()

        # ArmAssist moves itself every aa_tstep
        self.aa_tstep = aa_tstep

        # PI controller sends a new torque to ArmAssist every aa_pic_tstep
        self.aa_pic_tstep = aa_pic_tstep

        dtype = np.dtype('float64')

        self.bf = shared_np_mat(dtype, (3, 1))      # body frame
        self.wf = shared_np_mat(dtype, (3, 1))      # world frame
        self.bf_dot = shared_np_mat(dtype, (3, 1))  # body frame "dot"
        self.wf_dot = shared_np_mat(dtype, (3, 1))  # world frame "dot"
        self.torque = shared_np_mat(dtype, (3, 1))  # torque
        self.ex_f = shared_np_mat(dtype, (3, 1))    # external force

        self.active = mp.Value('b', 0)
        self.lock = mp.Lock()

        ### pic
        # if needed, edit to use shared memory for these variables too
        self.KP = KP
        self.TI = TI
        self.i_error = np.mat(np.zeros((TI.shape[1], 1)))

        self.wf_dot_ref = shared_np_mat(dtype, (3, 1))
        ## pic

        self.t_print = t_print

    def run(self):
        print 'starting ArmAssist'
        self.active.value = 1
        t_start = time.time()
        t_simulated_aa = 0  # how much time has been simulated for ArmAssist
        t_simulated_aa_pic = 0  # how much time has been simulated for ArmAssist PIC

        # prints its clock every t_print secs
        next_t_print = self.t_print
        
        while self.active.value == 1:
            # how much real time has elapsed
            t_elapsed = time.time() - t_start

            t = time.time()

            if t_elapsed - t_simulated_aa >= self.aa_tstep:
                self._step()
                t_simulated_aa += self.aa_tstep

            if t_elapsed - t_simulated_aa_pic >= self.aa_pic_tstep:
                self._pic_step()
                t_simulated_aa_pic += self.aa_pic_tstep

            t_sim = time.time() - t
            if t_sim < self.aa_tstep:
                time.sleep((self.aa_tstep - t_sim))  # TODO -- divide by 2?

            if t_elapsed > next_t_print:
                print 'ArmAssist at t=%3.3f s' % t_elapsed
                next_t_print += self.t_print

    def _step(self):
        with self.lock:
            self.bf[:, :], self.bf_dot[:, :] = rungekutta_body_frame(self.torque.copy(), self.bf.copy(), self.ex_f, self.aa_tstep)
            self.wf[:, :], self.wf_dot[:, :] = rungekutta_world_frame(self.wf.copy(), self.bf.copy(), self.aa_tstep)

    def stop(self):
        self.active.value = 0

    def get_state(self):
        state = dict()
        with self.lock:
            state['bf'] = self.bf.copy()
            state['wf'] = self.wf.copy()
            state['bf_dot'] = self.bf_dot.copy()
            state['wf_dot'] = self.wf_dot.copy()

        # return xy values in units of cm and cm/s, not m and m/s
        state['bf'][0:2] *= m_to_cm
        state['wf'][0:2] *= m_to_cm
        state['bf_dot'][0:2] *= m_to_cm
        state['wf_dot'][0:2] *= m_to_cm

        return state

    # IMPORTANT: only use to set starting position/orientation
    def _set_wf(self, wf):
        wf[0:2] *= cm_to_m
        with self.lock:
            self.wf[:, :] = np.mat(wf).reshape((3, 1))

    def set_torque(self, torque):
        self.torque[:, :] = torque
        # with self.lock:
        #     self.torque[:, :] = torque

    def set_external_force(self, ex_f):
        with self.lock:
            self.ex_f[:, :] = ex_f


    ## PIC methods
    def _pic_step(self):
        with self.lock:
            # state = self.get_state()
            
            # # PI controller assumes units of m, m/s, but get_state() returns 
            # # x, y values in units of cm, cm/s
            # state['bf'][0:2] *= cm_to_m
            # state['wf'][0:2] *= cm_to_m
            # bf = state['bf']
            # wf = state['wf']
            bf = self.bf.copy()
            wf = self.wf.copy()

            wf_2 = float(wf[2])  # necessary because w[2] returns a 1x1 matrix
            tmp = np.mat([[cos(wf_2),  sin(wf_2), 0], 
                          [-sin(wf_2), cos(wf_2), 0], 
                          [0,          0,         1]])
            bf_ref = tmp * self.wf_dot_ref

            # PI control
            error = bf_ref - bf  # error between desired and actual
            self.i_error += self.aa_pic_tstep * error # integration of error
            torque = self.KP * (error + self.TI*self.i_error)

            self.set_torque(torque)

    def update_reference(self, wf_dot_ref):
        wf_dot_ref[0:2] *= cm_to_m
        with self.lock:
            self.wf_dot_ref[:, :] = wf_dot_ref
