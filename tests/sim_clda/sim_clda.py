#!/usr/bin/python
"""
Simulation of CLDA control task!
"""
## Imports
from __future__ import division
import os
import optparse
import numpy as np

from riglib.bmi import kfdecoder, clda
import riglib.bmi

import multiprocessing as mp
from tasks import bmitasks, manualcontrol
import utils
from scipy.io import loadmat

import numpy as np
from numpy.random import poisson, rand
from utils import normalize
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import os
from numpy import *

from scipy.integrate import trapz, simps
import utils
import statsmodels.api as sm
from riglib.experiment.features import Autostart
from riglib.bmi.sim_neurons import CosEnc
import time

reload(kfdecoder)
reload(clda)
reload(riglib.bmi)
reload(riglib.bmi.train)


### Constants
DT = 0.1
COLORS = {}

COLORS['black']           = (0,   0,   0)
COLORS['white']           = (255, 255, 255)
COLORS['red']             = (255, 0,   0)
COLORS['green']           = (0,   255, 0)
COLORS['blue']            = (0,   0,   255)
COLORS['yellow']          = (255, 255, 0)
COLORS['cyan']            = (0,   255, 255)
GAME_COLORS               = {}
GAME_COLORS['background'] = COLORS['black']
GAME_COLORS['center']     = COLORS['cyan']
GAME_COLORS['target']     = COLORS['cyan']
GAME_COLORS['cursor']     = COLORS['yellow']
GAME_COLORS['sector']     = COLORS['red']
GAME_COLORS['text']       = COLORS['white']

parser = optparse.OptionParser()
parser.add_option("--show", action="store_true", dest="show", default=False)
(options, args) = parser.parse_args()

# Task parameters
m_to_mm = 1000
m_to_cm = 100
cm_to_m = 0.01

class PyGame():
    def __init__(self, input_device='mouse', win_res=300, show=False, 
        interactive=False):
        """
        """
        self.show = show
        self.interactive = interactive
        if self.show: 
            import pygame
            import pygame.locals as pl
            pygame.init()
            #pygame.mouse.set_visible(False)
            self.screen = pygame.display.set_mode((win_res, win_res))
            self.background = pygame.Surface(self.screen.get_size()).convert()
            self.background.fill(GAME_COLORS['background'])
        else:
            win_res = 1

        self.size = win_res
        self.scale = win_res/2
        self.input_device = input_device

        if input_device == 'joystick':
            # initialize main joystick device system
            pygame.joystick.init()

            # create and initialize a joystick instance
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

class CenterOut(PyGame):
    def __init__(self, n_trials=50, r_ctr=0.017, 
        r_targ=0.017, r_cursor=0.005, t_ctrhold=0.5, t_reachtarg=10,
        t_targhold=0.5, interactive=True, 
        workspace_ll= np.array([-0.07,0.06]), workspace_size=0.20, 
        workspace_targets = np.array([[ 0.1029723 ,  0.16483756],
               [ 0.08246977,  0.21433503],
               [ 0.0329723 ,  0.23483756],
               [-0.01652517,  0.21433503],
               [-0.0370277 ,  0.16483756],
               [-0.01652517,  0.11534008],
               [ 0.0329723 ,  0.09483756],
               [ 0.08246977,  0.11534008]]), 
        workspace_ctr=np.array([0.03297305, 0.16475039]),     
        target_list_fname='targ_list_50_trials.mat', 
        input_device='mouse', show=True):
        """ 
        Initialize center-out game
        """
        # game states
        self.GOTOCTR   = GOTOCTR  = 0
        self.HOLDCTR   = HOLDCTR  = 1
        self.GOTOTARG  = GOTOTARG = 2
        self.HOLDTARG  = HOLDTARG = 3

        ## Initialize parent Pygame
        self.show = show
        PyGame.__init__(self, input_device=input_device, show=self.show)

        ## store parameters
        self.n_trials       = n_trials 
        self.n_targs        = workspace_targets.shape[0] 
        self.r_ctr          = r_ctr
        self.r_targ         = r_targ
        self.t_ctrhold      = t_ctrhold
        self.t_reachtarg    = t_reachtarg
        self.t_targhold     = t_targhold
        self.interactive    = interactive
        self.workspace_ll = workspace_ll
        self.workspace_size = workspace_size
        self.workspace_ur = workspace_ll + workspace_size
        self.pix_per_m = self.size/workspace_size
        self.r_ctr_pix = int(r_ctr*self.pix_per_m)
        self.r_targ_pix = int(r_targ*self.pix_per_m)
        self.r_cursor_pix = int(r_cursor*self.pix_per_m)
        self.target_list_fname = target_list_fname
        
        self.horiz_min = self.workspace_ll[0]
        self.horiz_max = self.workspace_ur[0]
        self.vert_min = self.workspace_ll[1]
        self.vert_max = self.workspace_ur[1]

        self.target_positions = workspace_targets
        self.center = workspace_ctr 

        # Initialize Center-out game using parameters
        if os.path.exists(self.target_list_fname):
            self.target_stack = list(loadmat(self.target_list_fname)['target_stack'].reshape(-1))
        else:
            print "regenerating target list!"
            targets = np.random.randint(0, self.n_targs, self.n_trials)
            targets[0:self.n_targs] = np.arange(min(self.n_targs, self.n_trials))
            self.target_stack = list(targets)
            try:
                savemat(self.target_list_fname, {'target_stack':self.target_stack})
            except:
                pass

        self.cursor = np.array([0, 0])
        self.cursor_vel = np.array([0, 0])

        self.state = self.GOTOCTR

        self.time_elapsed = 0
        self.message = None

        self.targ_ind = None
        self.target = self.center 
        self.verbose = False
        
        if self.verbose:
            print "Number of targets: %d" % self.n_targs
            print "Number of trials: %d" % self.n_trials

        ## Event code list 
        self.event_codes = [ ]
        self.show_center = True
        self.show_target = False

    def pos2pix(self, kfpos):
        # rescale the cursor position to (0,1)
        norm_workspace_pos = (kfpos - self.workspace_ll)/self.workspace_size

        # multiply by the workspace size in pixels 
        pix_pos = self.size*norm_workspace_pos

        # flip y-coordinate
        pix_pos[1] = self.size - pix_pos[1]

        # cast to integer
        pix_pos = np.array(pix_pos, dtype=int) 
        return pix_pos

    def move_cursor(self, new_pos, run_fsm=True):
        """Move cursor position, and update game state and pygame screen.
        """
        
        self.cursor = new_pos #np.clip(self.cursor + diff, -1, 1)
        self.threshold_output()
        if self.show:
            self.draw()

    def threshold_output(self):
        self.cursor[0] = max( self.cursor[0], self.horiz_min )
        self.cursor[0] = min( self.cursor[0], self.horiz_max )
        self.cursor[1] = max( self.cursor[1], self.vert_min )
        self.cursor[1] = min( self.cursor[1], self.vert_max )
        
    def draw(self):
        """Update pygame screen.
        """
        import pygame
        self.screen.blit(self.background, (0, 0))

        #if self.state in (self.GOTOCTR, self.HOLDCTR):
        if self.show_center:
            pygame.draw.circle(self.screen, GAME_COLORS['center'], 
                self.pos2pix(self.center), self.r_ctr_pix)

        #if self.state in (self.GOTOTARG, self.HOLDTARG):
        if self.show_target:
            pygame.draw.circle(self.screen, GAME_COLORS['target'], 
                self.pos2pix(self.target), self.r_targ_pix)

        pygame.draw.circle(self.screen, GAME_COLORS['cursor'], 
            self.pos2pix(self.cursor), self.r_cursor_pix)
        if self.verbose:
            print "cursor position in pixels:"
            print self.pos2pix(self.cursor)

        pygame.display.update()

### Feedback controller
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

### Neural encoder
## Initialze encoder (reload from file)
print "creating ensemble ...."
ensemble_fname = 'test_ensemble.mat'
encoder = CosEnc(fname=ensemble_fname, return_ts=True)
n_neurons = encoder.n_neurons
units = encoder.get_units()

# Initialize input device
input_device = CenterOutCursorGoal(angular_noise_var=0.13)

if 0:
    center = np.array([0.0042056, 0.15983523])
    targets = np.array([
        [ 0.0692056 ,  0.0502056 ,  0.0042056 , -0.0417944 , -0.0607944 , -0.0417944,  0.0042056 ,  0.0502056 ],
        [ 0.15983523,  0.20573523,  0.22483523,  0.20573523,  0.15983523, 0.11393523,  0.09483523,  0.11393523]]).T
else:
    center = np.zeros(2)
    pi = np.pi
    targets = 0.065*np.vstack([[np.cos(pi/4*k), np.sin(pi/4*k)] for k in range(8)])
workspace_ll = center + np.array([-0.1, -0.1])
workspace_ur = center + np.array([0.1, 0.1])

def target_seq_generator(n_targs, n_trials):
    target_inds = np.random.randint(0, n_targs, n_trials)
    target_inds[0:n_targs] = np.arange(min(n_targs, n_trials))
    target_stack = list(targets)
    k = 0
    while k < n_trials:
        targ = m_to_cm*targets[target_inds[k], :]
        yield np.array([[center[0], 0, center[1]],
                        [targ[0], 0, targ[1]]]).T
        #yield np.array([m_to_mm*targets[target_inds[k], 0], 0, m_to_mm*targets[target_inds[k], 1]])
        k += 1

class FakeHDF():
    def __init__(self, *args, **kwargs):
        pass

    def sendMsg(self, msg):
        pass

    def __setitem__(self, key, value):
        pass

#trial_types = [np.array([[center[0], 0, center[1]], [targets[k,0], 0, targets[k,1]]]).T for k in range(8)]
class SimCLDAControl(bmitasks.CLDAControl, Autostart):
    #trial_types = trial_types
    update_rate = 0.1
    def __init__(self, *args, **kwargs):
        self.batch_time = 10.0
        self.half_life  = 20.0
        super(SimCLDAControl, self).__init__(*args, **kwargs)

        self.origin_hold_time = 0.250
        self.terminus_hold_time = 0.250
        self.origin_size = 1.7
        self.terminus_size = 1.7
        self.hdf = FakeHDF()
        self.task_data = FakeHDF()
        self.start_time = 0.
        self.loop_counter = 0

    def init(self):
        # BMI bounding box
        horiz_min = m_to_mm * workspace_ll[0]
        horiz_max = m_to_mm * workspace_ur[0]
        vert_min  = m_to_mm * workspace_ll[1]
        vert_max  = m_to_mm * workspace_ur[1]
        
        bounding_box = np.array([horiz_min, vert_min]), np.array([horiz_max, vert_max])
        states = ['hand_px', 'hand_pz', 'hand_vx', 'hand_vz', 'offset']
        #states = ['hand_px', 'hand_py', 'hand_pz', 'hand_vx', 'hand_vy', 'hand_vz', 'offset']
        states_to_bound = ['hand_px', 'hand_pz']
        neuron_driving_states = ['hand_vx', 'hand_vz', 'offset']
        drives_neurons = np.array([x in neuron_driving_states for x in states])
        
        stochastic_states = ['hand_vx', 'hand_vz']
        is_stochastic = np.array([x in stochastic_states for x in states])

        self.decoder = riglib.bmi.train._train_KFDecoder_2D_sim(is_stochastic, 
            drives_neurons, units, bounding_box, states_to_bound, include_y=False)
        self.decoder.kf.C /= m_to_mm
        self.decoder.kf.W *= m_to_mm**2

        ## Instantiate AdaptiveBMI
        super(SimCLDAControl, self).init()

        self.wait_time = 0
        self.pause = False

    def _test_penalty_end(self, ts):
        return True
        
    def screen_init(self):
        target_radius = self.terminus_size * cm_to_m
        center_radius = self.origin_size * cm_to_m
        cursor_radius = self.cursor.radius * cm_to_m
        t_ctrhold = 0.250
        t_reachtarg = 10
        t_targhold = 0.250
        ##center = np.array([0.0042056, 0.15983523])
        ##targets = np.array([
        ##    [ 0.0692056 ,  0.0502056 ,  0.0042056 , -0.0417944 , -0.0607944 , -0.0417944,  0.0042056 ,  0.0502056 ],
        ##    [ 0.15983523,  0.20573523,  0.22483523,  0.20573523,  0.15983523, 0.11393523,  0.09483523,  0.11393523]]).T
        ##workspace_ll = center+np.array([-0.1, -0.1])
        
        print "instantiating game..."
        self.game = CenterOut(
            show=options.show, r_ctr=m_to_mm*center_radius, r_targ=m_to_mm*target_radius,
            r_cursor=m_to_mm*cursor_radius, 
            t_ctrhold=t_ctrhold, t_reachtarg=t_reachtarg, t_targhold=t_targhold, 
            workspace_size=m_to_mm*0.20, workspace_ll=m_to_mm*workspace_ll,
            workspace_targets=m_to_mm*targets, workspace_ctr=m_to_mm*center, 
            n_trials=10
        )

    def show_origin(self, show=False):
        self.game.show_center = show

    def show_terminus(self, show=False):
        self.game.show_target = show
        if show:
            self.game.target = 10*np.array([self.terminus_target.xfm.move[0], self.terminus_target.xfm.move[2]])
            print self.game.target

    def redraw(self):
        pass

    def get_spike_ts(self):
        cursor_pos = [10*self.cursor.xfm.move[0], 10*self.cursor.xfm.move[2]]
        target_pos = self.target_xz
        ctrl    = input_device.get(cursor_pos, target_pos)
        ts_data = encoder(ctrl)
        return ts_data

    def _update(self, pt):
        super(SimCLDAControl, self)._update(pt)

    def get_time(self):
        return self.loop_counter * DT

    def run(self):
        '''
        Generic method to run the finite state machine of the task
        '''
        self.screen_init()
        self.set_state(self.state)
        while self.state is not None:
            if hasattr(self, "_while_%s"%self.state):
                getattr(self, "_while_%s"%self.state)()
            if hasattr(self, "_cycle"):
                self._cycle()
            
            for event, state in self.status[self.state].items():
                if hasattr(self, "_test_%s"%event):
                    if getattr(self, "_test_%s"%event)(self.get_time() - self.start_time):
                    #if getattr(self, "_test_%s"%event)(time.time() - self.start_time):
                        if hasattr(self, "_end_%s"%self.state):
                            getattr(self, "_end_%s"%self.state)()
                        self.trigger_event(event)
                        break;
            self.loop_counter += 1.

    def draw_world(self):
        print self.state
        if self.state == 'origin':
            self.game.show_origin = True
            self.game.show_target = False
        elif self.state == 'origin_hold':
            self.game.show_origin = True
            self.game.show_target = True
        elif self.state in ['terminus', 'terminus_hold']:
            self.game.show_origin = False
            self.game.show_target = True
        else:
            self.game.show_origin = False
            self.game.show_target = False
        cursor_pos = [10*self.cursor.xfm.move[0], 10*self.cursor.xfm.move[2]]
        self.game.move_cursor(cursor_pos, run_fsm=False)
        #time.sleep(1./10)
        #time.sleep(1./60)
        #time.sleep(self.update_rate)

    #def update_target_location(self):
    #    self.target_xz = self.game.get_target()

    #def _rescale_bmi_state(self, decoded_state):
    #    return decoded_state

    ## def update_cursor(self):
    ##     # Runs every loop
    ##     self.update_target_location()
    ##     self.update_learn_flag()
    ##     
    ##     ts_data = self.get_neural_data()

    ##     # Get the decoder output
    ##     decoded_state, update_flag = self.bmi_system(ts_data, self.target_xz,
    ##         self.state, task_data=self.task_data, assist_level=self.assist_level,
    ##         learn_flag=self.learn_flag)
    ##     # The update flag is true if the decoder parameters were updated on this
    ##     # iteration. If so, save an update message to the file.
    ##     if update_flag:
    ##         #send msg to hdf file
    ##         self.hdf.sendMsg("update_bmi")

    ##     # Remember that decoder is only decoding in 2D, y value is set to 0
    ##     pt = np.array([decoded_state[0], 0, decoded_state[1]])
    ##     #pt = np.array([0.1*decoded_state[0], 0, 0.1*decoded_state[1]])
    ##     # Save cursor location to file
    ##     self.task_data['cursor'] = pt[:3]
    ##     # Update cursor location
    ##     self._update(pt[:3])   
    ##     # Write to screen
    ##     self.draw_world()


    ## def update_cursor(self):
    ##     # Runs every loop
    ##     self.update_target_location()

    ##     ts_data = self.get_neural_data()

    ##     # Get the decoder output
    ##     decoded_state, update_flag = self.bmi_system(ts_data, self.target_xz, '',
    ##         task_data=None, assist_level=self.assist_level,
    ##         learn_flag=self.learn_flag)

    ##     # The update flag is true if the decoder parameters were updated on this
    ##     # iteration. If so, save an update message to the file.
    ##     if update_flag:
    ##         #send msg to hdf file
    ##         self.hdf.sendMsg("update_bmi")

    ##     # Remember that decoder is only decoding in 2D, y value is set to 0
    ##     pt = np.array([decoded_state[0], 0, decoded_state[1]])
    ##     #pt = np.array([0.1*decoded_state[0], 0, 0.1*decoded_state[1]])
    ##     # Save cursor location to file
    ##     self.task_data['cursor'] = pt[:3]
    ##     # Update cursor location
    ##     self._update(pt[:3])   
    ##     # Write to screen
    ##     self.draw_world()

    ##     #self.game.move_cursor([decoded_state[0], decoded_state[1]])


gen = target_seq_generator(8, 1000)
task = SimCLDAControl(gen)
task.init()
task.run()

# for k in range(10000):
#     cursor     = game.cursor
#     ctrl       = input_device.get(game, None, gain=m_to_mm*0.15)
#     target_pos = game.get_target()
# 
#     # generate synthetic neural activity from 'ctrl'
#     spike_obs = encoder(ctrl/m_to_mm)
# 
#     if spike_obs.dtype == kfdecoder.python_plexnet_dtype:
#         spike_counts = bmi.decoder.bin_spikes(spike_obs)
#         assert sum(spike_counts) == len(spike_obs)
#     # decode neural observation
#     #target_pos = np.hstack([target_pos[0], 0, target_pos[1]])
#     bmi(spike_obs, target_pos, '', assist_level=0)
# 
#     # move cursor
#     try:
#         #game.move_cursor(decoder['px', 'py'])
#         game.move_cursor(decoder['hand_px', 'hand_pz'])
#     except:
#         break
#     game.rotate_vel(decoder['hand_vx', 'hand_vz'])
# 
# bmi.__del__()
