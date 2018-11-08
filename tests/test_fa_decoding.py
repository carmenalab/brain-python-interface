from features.simulation_features import SimKFDecoderSup, SimCosineTunedEnc, SimTime, SimFAEnc
from features.hdf_features import SaveHDF
from features.generator_features import Autostart
from riglib.stereo_opengl.window import FakeWindow
from riglib.bmi.state_space_models import StateSpaceEndptVel2D, State, offset_state
from riglib.bmi.feedback_controllers import LQRController
from riglib import experiment

from tasks.bmimultitasks import BMIControlMulti
from tasks import manualcontrolmultitasks

import plantlist

import numpy as np
import os, shutil, pickle
import pickle
import time, datetime

from riglib.bmi.state_space_models import StateSpaceEndptVel2D, StateSpaceEndptVelY
from riglib.bmi import feedback_controllers

class SuperSimpleEndPtAssister(object):
    '''
    Constant velocity toward the target if the cursor is outside the target. If the
    cursor is inside the target, the speed becomes the distance to the center of the
    target divided by 2.
    '''
    def __init__(self, *args, **kwargs):
        '''    Docstring    '''
        self.decoder_binlen = 0.1
        self.assist_speed = 5.
        self.target_radius = 2.

    def calc_next_state(self, current_state, target_state, mode=None, **kwargs):
        '''    Docstring    '''
        
        cursor_pos = np.array(current_state[0:3,0]).ravel()
        target_pos = np.array(target_state[0:3,0]).ravel()
        decoder_binlen = self.decoder_binlen
        speed = self.assist_speed * decoder_binlen
        target_radius = self.target_radius

        diff_vec = target_pos - cursor_pos 
        dist_to_target = np.linalg.norm(diff_vec)
        dir_to_target = diff_vec / (np.spacing(1) + dist_to_target)
        
        if dist_to_target > target_radius:
            assist_cursor_pos = cursor_pos + speed*dir_to_target
        else:
            assist_cursor_pos = cursor_pos + speed*diff_vec/2

        assist_cursor_vel = (assist_cursor_pos-cursor_pos)/decoder_binlen
        x_assist = np.hstack([assist_cursor_pos, assist_cursor_vel, 1])
        x_assist = np.mat(x_assist.reshape(-1,1))
        return x_assist

#SimCosineTunedEnc
class SimVFB(Autostart, SimTime, FakeWindow, SimKFDecoderSup, SimFAEnc, BMIControlMulti):
    sequence_generators = ['centerout_3D_discrete','centerout_2D_discrete','centerout_Y_discrete']

    def __init__(self, ssm, init_C_func, *args, **kwargs):
        self.assist_level = (1, 1)
        self.n_neurons = 20
        #OFC FB CTRL   from BMIControlMulti
        # F = np.eye(7)
        # decoding_rate = 10.
        # B = np.mat(np.vstack([np.zeros([3,3]), np.eye(3)*1000*1./decoding_rate, np.zeros(3)]))
        # B = np.hstack((np.zeros((7, 3)), B, np.zeros((7, 1)) ))
        # self.fb_ctrl = feedback_controllers.LinearFeedbackController(A=B, B=B, F=F)
        self.fb_ctrl = SuperSimpleEndPtAssister()
        self.sim_C = np.zeros((self.n_neurons, ssm.n_states))

        # self.A, self.B, _ = ssm.get_ssm_matrices()
        # self.Q = np.mat(np.diag([1., 1., 1., 5, 5, 5, 0]))
        # self.R = 1e6 * np.mat(np.diag([1., 1., 1.]))
        # self.fb_ctrl = LQRController(self.A, self.B, self.Q, self.R)

        self.plant = plantlist.plantlist[self.plant_type]
        #self.sim_C = init_C_func(self.sim_C)
        self.ssm=ssm

        super(SimVFB, self).__init__(*args, **kwargs)


    @classmethod
    def xz_sim(cls, C):
        assert C.shape[0]==6
        nS = C.shape[1] #Number of states
        #Neuron 0 is untuned, so keep row 1 all zeros
        C[0, :] = np.zeros((nS))
        C[0, -1] = 15
        #Neuron 1 is tuned to +z vel
        C[1, :] = np.array([0, 0, 0, 0, 0, 5, 15])
        #Neurons 2, 3 are tuned to +x vel:
        C[2, :] = np.array([0, 0, 0, 5, 0, 0, 15])
        C[3, :] = np.array([0, 0, 0, 5, 0, 0, 15])

        #Neurons 4, 5 are tuned to same noise:
        C[4, :] = np.array([0, 0, 0, 0, 0, 0, 30])
        C[5, :] = np.array([0, 0, 0, 0, 0, 0, 30])
        return C

    @classmethod
    def y_sim(cls, C):
        assert C.shape[0]==6
        nS = C.shape[1] #Number of states
        #Neuron 0 is untuned, so keep row 1 all zeros
        C[0, :] = np.zeros((nS))
        C[0, -1] = 15
        #Neuron 1 is tuned to +z vel
        C[1, :] = np.array([0, 0, 0, 0, -10, 5, 15])
        #Neurons 2, 3 are tuned to +x vel:
        C[2, :] = np.array([0, 0, 0, 5, 5, 0, 15])
        C[3, :] = np.array([0, 0, 0, 5, 10, 0, 15])

        #Neurons 4, 5 are tuned to same noise:
        C[4, :] = np.array([0, 0, 0, 0, 0, 0, 30])
        C[5, :] = np.array([0, 0, 0, 0, 0, 0, 30])
        return C

    def _cycle(self, *args, **kwargs):
        super(SimVFB, self)._cycle(*args, **kwargs)

    @staticmethod
    def centerout_Y_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12,-12,12),
        distance=10):

        theta = []
        for i in range(nblocks):
            temp1 = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp1)
            theta = theta + list(temp1)
        theta = np.vstack(theta)
        
        x = z = np.zeros((len(theta), 1))
        y = distance*np.cos(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.hstack([x, y, z])
        
        return pairs

    @staticmethod
    def centerout_3D_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12,-12,12),
        distance=10):
        '''

        Generates a sequence of 3D target pairs with the first target
        always at the origin.

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between the targets in a pair.

        Returns
        -------
        pairs : [nblocks*ntargets x 2 x 3] array of pairs of target locations


        '''
        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            for t in temp:
                temp1 = np.array([t]*ntargets)
                temp2 = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
                np.random.shuffle(temp2)
                theta = theta + list(zip(temp1, temp2))
        np.random.shuffle(theta)
        theta = np.vstack(theta)

        y_x = np.tan(theta[:,0])
        z = distance*np.cos(theta[:,1])
        x = np.sqrt((distance**2 - (z**2))/(1+(y_x**2)))
        x[:int(.5*len(x))] = x[:int(.5*len(x))] * - 1
        y = x*y_x

        ix = np.arange(len(x))  
        np.random.shuffle(ix)
        x = x[ix]
        y = y[ix]
        z = z[ix] 

        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs


#Playback trajectories
def main_xz(session_length):
    ssm_xz = StateSpaceEndptVel2D()
    Task = experiment.make(SimVFB, [SaveHDF])
    #targets = SimVFB.centerout_3D_discrete()
    targets = manualcontrolmultitasks.ManualControlMulti.centerout_2D_discrete()
    task = Task(ssm_xz, SimVFB.xz_sim, targets, plant_type='cursor_14x14', session_length=session_length)
    task.run_sync()
    return task

def main_Y(session_length):
    ssm_y = StateSpaceEndptVelY()
    Task = experiment.make(SimVFB, [SaveHDF])
    targets = SimVFB.centerout_Y_discrete()
    #targets = manualcontrolmultitasks.ManualControlMulti.centerout_2D_discrete()
    task = Task(ssm_y, SimVFB.y_sim, targets, plant_type='cursor_14x14', session_length=session_length)
    task.run_sync()
    return task


def save_stuff(task, suffix=''):
    enc = task.encoder
    task.decoder.save()
    enc.corresp_dec = task.decoder

    #Save task info
    ct = datetime.datetime.now()
    pnm = os.path.expandvars('$FA_GROM_DATA/sims/enc'+ ct.strftime("%m%d%y_%H%M") + suffix + '.pkl')
    pickle.dump(enc, open(pnm,'wb'))

    #Save HDF file
    new_hdf = pnm[:-4]+'.hdf'
    f = open(task.h5file.name)
    f.close()

    #Wait 
    time.sleep(1.)

    #Wait after HDF cleaned up
    task.cleanup_hdf()
    time.sleep(1.)

    #Copy temp file to actual desired location
    shutil.copy(task.h5file.name, new_hdf)
    f = open(new_hdf)
    f.close()

    #Return filename
    return pnm