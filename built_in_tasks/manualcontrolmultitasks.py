'''
Base tasks for generic point-to-point reaching
'''

import numpy as np
from collections import OrderedDict
import time

from riglib import reward
from riglib.experiment import traits, Sequence

from riglib.stereo_opengl.window import Window, FPScontrol, WindowDispl2D
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere, Cube
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex

from plantlist import plantlist

from riglib.stereo_opengl import ik
import os

import math
import traceback

####### CONSTANTS
sec_per_min = 60.0
RED = (1,0,0,.5)
GREEN = (0,1,0,0.5)
GOLD = (1., 0.843, 0., 0.5)
mm_per_cm = 1./10

from target_graphics import *

target_colors = {
"yellow": (1,1,0,0.75),
"magenta": (1,0,1,0.75),
"purple":(0.608,0.188,1,0.75),
"dodgerblue": (0.118,0.565,1,0.75),
"teal":(0,0.502,0.502,0.75),
"olive":(0.420,0.557,0.137,.75),
"juicyorange": (1,0.502,0.,0.75),
"hotpink":(1,0.0,0.606,.75),
"lightwood": (0.627,0.322,0.176,0.75),
"elephant":(0.409,0.409,0.409,0.5),
"green":(0., 1., 0., 0.5)}


class ManualControlMulti(Sequence, Window):
    '''
    This is an improved version of the original manual control tasks that includes the functionality
    of ManualControl, ManualControl2, and TargetCapture all in a single task. This task doesn't
    assume anything about the trial structure of the task and allows a trial to consist of a sequence
    of any number of sequential targets that must be captured before the reward is triggered. The number
    of targets per trial is determined by the structure of the target sequence used.
    '''

    background = (0,0,0,1)
    cursor_color = (.5,0,.5,1)

    plant_type = traits.OptionsList(*plantlist, desc='', bmi3d_input_options=list(plantlist.keys()))

    starting_pos = (5, 0, 5)

    status = dict(
        wait = dict(start_trial="target", stop=None),
        target = dict(enter_target="hold", timeout="timeout_penalty", stop=None),
        hold = dict(leave_early="hold_penalty", hold_complete="targ_transition", stop=None),
        targ_transition = dict(trial_complete="reward",trial_abort="wait", trial_incomplete="target", stop=None),
        timeout_penalty = dict(timeout_penalty_end="targ_transition", stop=None),
        hold_penalty = dict(hold_penalty_end="targ_transition", stop=None),
        reward = dict(reward_end="wait")
    )
    trial_end_states = ['reward', 'timeout_penalty', 'hold_penalty']

    #initial state
    state = "wait"

    target_color = (1,0,0,.5)
    target_index = -1 # Helper variable to keep track of which target to display within a trial
    tries = 0 # Helper variable to keep track of the number of failed attempts at a given trial.
    
    cursor_visible = False # Determines when to hide the cursor.
    no_data_count = 0 # Counter for number of missing data frames in a row
    scale_factor = 3.0 #scale factor for converting hand movement to screen movement (1cm hand movement = 3.5cm cursor movement)

    limit2d = 1

    sequence_generators = ['centerout_2D_discrete', 'centerout_2D_discrete_offset', 'point_to_point_3D', 'centerout_3D', 'centerout_3D_cube', 'centerout_2D_discrete_upper','centerout_2D_discrete_rot', 'centerout_2D_discrete_multiring',
        'centerout_2D_discrete_randorder', 'centeroutback_2D', 'centeroutback_2D_farcatch', 'centeroutback_2D_farcatch_discrete',
        'outcenterout_2D_discrete', 'outcenter_2D_discrete', 'rand_target_sequence_3d', 'rand_target_sequence_2d', 'rand_target_sequence_2d_centerout',
        'rand_target_sequence_2d_partial_centerout', 'rand_multi_sequence_2d_centerout2step', 'rand_pt_to_pt',
        'centerout_2D_discrete_far', 'centeroutback_2D_v2','centerout_2D_discrete_eyetracker_calibration']
    is_bmi_seed = True
    _target_color = RED


    # Runtime settable traits
    reward_time = traits.Float(.5, desc="Length of juice reward")
    target_radius = traits.Float(2, desc="Radius of targets in cm")
    
    hold_time = traits.Float(.2, desc="Length of hold required at targets")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    timeout_time = traits.Float(10, desc="Time allowed to go between targets")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for timeout error")
    max_attempts = traits.Int(10, desc='The number of attempts at a target before\
        skipping to the next one')
    # session_length = traits.Float(0, desc="Time until task automatically stops. Length of 0 means no auto stop.")
    marker_num = traits.Int(14, desc="The index of the motiontracker marker to use for cursor position")
    # NOTE!!! The marker on the hand was changed from #0 to #14 on
    # 5/19/13 after LED #0 broke. All data files saved before this date
    # have LED #0 controlling the cursor.

    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')    
    plant_type_options = list(plantlist.keys())
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    cursor_radius = traits.Float(.5, desc="Radius of cursor")
    
    def __init__(self, *args, **kwargs):
        super(ManualControlMulti, self).__init__(*args, **kwargs)
        self.cursor_visible = True

        # Initialize the plant
        if not hasattr(self, 'plant'):
            self.plant = plantlist[self.plant_type]
        self.plant_vis_prev = True

        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            target1 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)
            target2 = VirtualCircularTarget(target_radius=self.target_radius, target_color=self._target_color)

            self.targets = [target1, target2]
            for target in self.targets:
                for model in target.graphics_models:
                    self.add_model(model)
        
        # Initialize target location variable
        self.target_location = np.array([0, 0, 0])

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

    def init(self):
        self.add_dtype('target', 'f8', (3,))
        self.add_dtype('target_index', 'i', (1,))
        super(ManualControlMulti, self).init()

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen. Runs 60x per second.
        '''
        self.task_data['target'] = self.target_location.copy()
        self.task_data['target_index'] = self.target_index

        ## Run graphics commands to show/hide the plant if the visibility has changed
        if self.plant_type != 'CursorPlant':
            if self.plant_visible != self.plant_vis_prev:
                self.plant_vis_prev = self.plant_visible
                self.plant.set_visibility(self.plant_visible)
                # self.show_object(self.plant, show=self.plant_visible)

        self.move_effector()

        ## Save plant status to HDF file
        plant_data = self.plant.get_data_to_save()
        for key in plant_data:
            self.task_data[key] = plant_data[key]

        super(ManualControlMulti, self)._cycle()
        
    def move_effector(self):
        ''' Sets the plant configuration based on motiontracker data. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        #get data from motion tracker- take average of all data points since last poll
        pt = self.motiondata.get()
        if len(pt) > 0:
            pt = pt[:, self.marker_num, :]
            conds = pt[:, 3]
            inds = np.nonzero((conds>=0) & (conds!=4))[0]
            if len(inds) > 0:
                pt = pt[inds,:3]
                #scale actual movement to desired amount of screen movement
                pt = pt.mean(0) * self.scale_factor
                #Set y coordinate to 0 for 2D tasks
                if self.limit2d: pt[1] = 0
                pt[1] = pt[1]*2
                # Return cursor location
                self.no_data_count = 0
                pt = pt * mm_per_cm #self.convert_to_cm(pt)
            else: #if no usable data
                self.no_data_count += 1
                pt = None
        else: #if no new data
            self.no_data_count +=1
            pt = None

        # Set the plant's endpoint to the position determined by the motiontracker, unless there is no data available
        if pt is not None:
            self.plant.set_endpoint_pos(pt)

    def run(self):
        '''
        See experiment.Experiment.run for documentation. 
        '''
        # Fire up the plant. For virtual/simulation plants, this does little/nothing.
        self.plant.start()
        try:
            super(ManualControlMulti, self).run()
        finally:
            self.plant.stop()

    ##### HELPER AND UPDATE FUNCTIONS ####
    def update_cursor_visibility(self):
        ''' Update cursor visible flag to hide cursor if there has been no good data for more than 3 frames in a row'''
        prev = self.cursor_visible
        if self.no_data_count < 3:
            self.cursor_visible = True
            if prev != self.cursor_visible:
            	self.show_object(self.cursor, show=True)
        else:
            self.cursor_visible = False
            if prev != self.cursor_visible:
            	self.show_object(self.cursor, show=False)

    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super(ManualControlMulti, self).update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120.), decimals=2)

    #### TEST FUNCTIONS ####
    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        return d <= (self.target_radius - self.cursor_radius)
        
    def _test_leave_early(self, ts):
        '''
        return true if cursor moves outside the exit radius
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target_location)
        rad = self.target_radius - self.cursor_radius
        return d > rad

    def _test_hold_complete(self, ts):
        return ts>=self.hold_time

    def _test_timeout(self, ts):
        return ts>self.timeout_time

    def _test_timeout_penalty_end(self, ts):
        return ts>self.timeout_penalty_time

    def _test_hold_penalty_end(self, ts):
        return ts>self.hold_penalty_time

    def _test_trial_complete(self, ts):
        return self.target_index==self.chain_length-1

    def _test_trial_incomplete(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries<self.max_attempts)

    def _test_trial_abort(self, ts):
        return (not self._test_trial_complete(ts)) and (self.tries==self.max_attempts)

    def _test_reward_end(self, ts):
        return ts>self.reward_time

    #### STATE FUNCTIONS ####
    def _parse_next_trial(self):
        self.targs = self.next_trial

    def _start_wait(self):
        super(ManualControlMulti, self)._start_wait()
        self.tries = 0
        self.target_index = -1
        #hide targets
        for target in self.targets:
            target.hide()

        self.chain_length = self.targs.shape[0] #Number of sequential targets in a single trial

    def _start_target(self):
        self.target_index += 1

        #move a target to current location (target1 and target2 alternate moving) and set location attribute
        target = self.targets[self.target_index % 2]
        self.target_location = self.targs[self.target_index]
        target.move_to_position(self.target_location)
        target.cue_trial_start()

    def _start_hold(self):
        #make next target visible unless this is the final target in the trial
        idx = (self.target_index + 1)
        if idx < self.chain_length: 
            target = self.targets[idx % 2]
            target.move_to_position(self.targs[idx])
    
    def _end_hold(self):
        # change current target color to green
        self.targets[self.target_index % 2].cue_trial_end_success()

    def _start_hold_penalty(self):
    	#hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1
    
    def _start_timeout_penalty(self):
    	#hide targets
        for target in self.targets:
            target.hide()

        self.tries += 1
        self.target_index = -1

    def _start_targ_transition(self):
        #hide targets
        for target in self.targets:
            target.hide()

    def _start_reward(self):
        #super(ManualControlMulti, self)._start_reward()
        self.targets[self.target_index % 2].show()

    #### Generator functions ####
    @staticmethod
    def point_to_point_3D(length=2000, boundaries=(-18,18,-10,10,-15,15), distance=10, chain_length=2):1

    @staticmethod
    def centerout_3D(length=1000, boundaries=(-18,18,-10,10,-15,15),distance=8):
        # Choose a random sequence of points on the surface of a sphere of radius
        # "distance"
        theta = np.random.rand(length)*2*np.pi
        phi = np.arccos(2*np.random.rand(length) - 1)
        x = distance*np.cos(theta)*np.sin(phi)
        y = distance*np.sin(theta)*np.sin(phi)
        z = distance*np.cos(theta)

        pairs = np.zeros([length,2,3])
        pairs[:,1,0] = x
        pairs[:,1,1] = y
        pairs[:,1,2] = z

        return pairs

    @staticmethod
    def centerout_3D_cube(length=1000, edge_length=8):
        '''
        Choose a random sequence of points on the surface of a sphere of radius
        "distance"
        '''
        coord = [-float(edge_length)/2, float(edge_length)/2]
        from itertools import product
        target_locs = [(x, y, z) for x, y, z in product(coord, coord, coord)]
        
        n_corners_in_cube = 8
        pairs = np.zeros([length, 2, 3])

        for k in range(length):
            pairs[k, 0, :] = np.zeros(3)
            pairs[k, 1, :] = target_locs[np.random.randint(0, n_corners_in_cube)]

        print(pairs.shape)
        return pairs

    @staticmethod
    def centerout_2D_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
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
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs
   
    @staticmethod
    def centerout_2D_discrete_offset(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=5,xoffset = -8, zoffset = 0, centeroffset = -8):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin (offset from center of screen).

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
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)+xoffset
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)+zoffset
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        pairs[:,0,:] = np.array([centeroffset, 0, 0])
        
        return pairs

    @staticmethod
    def centerout_2D_discrete_eyetracker_calibration(nblocks=100, ntargets=4, boundaries=(-18,18,-12,12),
        distance=10):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin. The order required by the eye-tracker calibration
        is Center, Left, Right, Up, Down. The sequence generator therefore
        displays 4- trial sequences in the following order:
        Center-Left (C-L), C-R, C-U, C-D.

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

        # Create a LRUD (Left Right Up Down) sequence of points on the edge of
        # a circle of radius "distance"

        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets) # ntargets = 4 --> shape of a +
            temp2 = np.array([temp[2],temp[0],temp[1],temp[3]]) # Left Right Up Down
            theta = theta + [temp2]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)

        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs

    @staticmethod
    def centerout_2D_discrete_upper(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):
        '''Same as centerout_2D_discrete, but rotates position of targets by 'rotate_deg'.
           For example, if you wanted only 1 target, but sometimes wanted it at pi and sometiems at 3pi/2,
             you could rotate it by 90 degrees'''
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, np.pi, np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs
    
    @staticmethod
    def centerout_2D_discrete_rot(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10,rotate_deg=0):
        '''Same as centerout_2D_discrete, but rotates position of targets by 'rotate_deg'.
           For example, if you wanted only 1 target, but sometimes wanted it at pi and sometiems at 3pi/2,
             you could rotate it by 90 degrees'''
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            temp = temp + (rotate_deg)*(2*np.pi/360)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)


        x = distance*np.cos(theta)
        y = np.zeros(len(theta))
        z = distance*np.sin(theta)
        
        pairs = np.zeros([len(theta), 2, 3])
        pairs[:,1,:] = np.vstack([x, y, z]).T
        
        return pairs

    @staticmethod
    def centerout_2D_discrete_multiring(n_blocks=100, n_angles=8, boundaries=(-18,18,-12,12),
        distance=10, n_rings=2):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
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
        target_set = []
        angles = np.arange(0, 2*np.pi, 2*np.pi/n_angles)
        distances = np.arange(0, distance + 1, float(distance)/n_rings)[1:]
        for angle in angles:
            for dist in distances:
                targ = np.array([np.cos(angle), 0, np.sin(angle)]) * dist
                target_set.append(targ)

        target_set = np.vstack(target_set)
        n_targets = len(target_set)
        

        periph_target_list = []
        for k in range(n_blocks):
            target_inds = np.arange(n_targets)
            np.random.shuffle(target_inds)
            periph_target_list.append(target_set[target_inds])

        periph_target_list = np.vstack(periph_target_list)

        
        pairs = np.zeros([len(periph_target_list), 2, 3])
        pairs[:,1,:] = periph_target_list#np.vstack([x, y, z]).T
        
        return pairs

    @staticmethod
    def centerout_2D_discrete_far(nblocks=100, ntargets=8, xmax=25, xmin=-25, zmin=-14, zmax=14, distance=10):
        target_angles = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
        target_pos = np.vstack([np.cos(target_angles), np.zeros_like(target_angles), np.sin(target_angles)]).T*distance
        target_pos = np.vstack([targ for targ in target_pos if targ[0] < xmax and targ[0] > xmin and targ[2] > zmin and targ[2] < zmax])

        from riglib.experiment.generate import block_random
        periph_targets_per_trial = block_random(target_pos, nblocks=nblocks)
        target_seqs = []
        for targ in periph_targets_per_trial:
            target_seqs.append(np.vstack([np.zeros(3), targ]))
        return target_seqs

    @staticmethod
    def centerout_2D_discrete_randorder(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=10):                                                                 
        '''                                                                           
                                                                                      
        Generates a sequence of 2D (x and z) target pairs with the first target       
        always at the origin, totally randomized instead of block randomized.         
                                                                                      
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
            np.random.shuffle(temp)                                                   
            theta = theta + [temp]                                                    
        theta = np.hstack(theta)                                                      
        np.random.shuffle(theta)                                                      
                                                                                      
                                                                                      
        x = distance*np.cos(theta)                                                    
        y = np.zeros(len(theta))                                                      
        z = distance*np.sin(theta)                                                    
                                                                                      
        pairs = np.zeros([len(theta), 2, 3])                                          
        pairs[:,1,:] = np.vstack([x, y, z]).T                                         
                                                                                      
        return pairs

    @staticmethod
    def centeroutback_2D(length, boundaries=(-18,18,-12,12), distance=8):
        '''
        Generates a sequence of 2D (x and z) center-edge-center target triplets.

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        theta = np.random.rand(length)*2*np.pi
        x = distance*np.cos(theta)
        z = distance*np.sin(theta)
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([length, 3, 3])
        targs[:,1,0] = x
        targs[:,1,2] = z
        
        return targs

    @staticmethod
    def centeroutback_2D_v2(length, boundaries=(-18,18,-12,12), distance=8):
        '''
        This fn exists purely for compatibility reasons?
        '''
        return centeroutback_2D(length, boundaries=boundaries, distance=distance)

    @staticmethod
    def centeroutback_2D_farcatch(length, boundaries=(-18,18,-12,12), distance=8, catchrate=.1):
        '''

        Generates a sequence of 2D (x and z) center-edge-center target triplets, with occasional
        center-edge-far edge catch trials thrown in.

        Parameters
        ----------
        length : int
            The number of target sets in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.
        catchrate: float
            The percent of trials that are far target catch trials. If distance*2

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations


        '''

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance" and a corresponding set of points on circle of raidus 2*distance
        theta = np.random.rand(length)*2*np.pi
        x = distance*np.cos(theta)
        z = distance*np.sin(theta)

        x2 = 2*x
        x2[np.nonzero((x2<boundaries[0]) | (x2>boundaries[1]))] = np.nan

        z2 = 2*z
        z2[np.nonzero((z2<boundaries[2]) | (z2>boundaries[3]))] = np.nan

        outertargs = np.zeros([length, 3])
        outertargs[:,0] = x2
        outertargs[:,2] = z2
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([length, 3, 3])
        targs[:,1,0] = x
        targs[:,1,2] = z

        # shuffle order of indices and select specified percent to use for catch trials
        numcatch = int(length*catchrate)
        shuffinds = np.array(list(range(length)))
        np.random.shuffle(shuffinds)
        replace = shuffinds[:numcatch]
        count = numcatch

        while np.any(np.isnan(outertargs[list(replace)])):
            replace = replace[~np.isnan(np.sum(outertargs[list(replace)],axis=1))]
            diff = numcatch - len(replace)
            new = shuffinds[count:count+diff]
            replace = np.concatenate((replace, new))
            count += diff

        targs[list(replace),2,:] = outertargs[list(replace)]

        return targs

    @staticmethod
    def centeroutback_2D_farcatch_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=8, catchrate=0):
        '''

        Generates a sequence of 2D (x and z) center-edge-center target triplets, with occasional
        center-edge-far edge catch trials thrown in.

        Parameters
        ----------
        length : int
            The number of target sets in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.
        catchrate: float
            The percent of trials that are far target catch trials. If distance*2

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations


        '''

       # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)

        x = distance*np.cos(theta)
        z = distance*np.sin(theta)

        # Choose a corresponding set of points on circle of radius 2*distance. Mark any points that
        # are outside specified boundaries with nans

        x2 = 2*x
        x2[np.nonzero((x2<boundaries[0]) | (x2>boundaries[1]))] = np.nan

        z2 = 2*z
        z2[np.nonzero((z2<boundaries[2]) | (z2>boundaries[3]))] = np.nan

        outertargs = np.zeros([nblocks*ntargets, 3])
        outertargs[:,0] = x2
        outertargs[:,2] = z2
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([nblocks*ntargets, 3, 3])
        targs[:,1,0] = x
        targs[:,1,2] = z

        # shuffle order of indices and select specified percent to use for catch trials
        numcatch = int(nblocks*ntargets*catchrate)
        shuffinds = np.array(list(range(nblocks*ntargets)))
        np.random.shuffle(shuffinds)
        replace = shuffinds[:numcatch]
        count = numcatch

        while np.any(np.isnan(outertargs[list(replace)])):
            replace = replace[~np.isnan(np.sum(outertargs[list(replace)],axis=1))]
            diff = numcatch - len(replace)
            new = shuffinds[count:count+diff]
            replace = np.concatenate((replace, new))
            count += diff

        targs[list(replace),2,:] = outertargs[list(replace)]

        return targs

    @staticmethod
    def outcenterout_2D_discrete(nblocks=100, ntargets=8, boundaries=(-18,18,-12,12),
        distance=8):
        '''

        Generates a sequence of 2D (x and z) center-edge-center target triplets, with occasional
        center-edge-far edge catch trials thrown in.

        Parameters
        ----------
        length : int
            The number of target sets in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.
        catchrate: float
            The percent of trials that are far target catch trials. If distance*2

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations


        '''

       # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(0, 2*np.pi, 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)

        x = distance*np.cos(theta)
        z = distance*np.sin(theta)
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([nblocks*ntargets, 3, 3])
        targs[:,0,0] = x
        targs[:,2,0] = x
        targs[:,0,2] = z
        targs[:,2,2] = z

        return targs

    @staticmethod
    def outcenter_2D_discrete(nblocks=100, ntargets=4, boundaries=(-18,18,-12,12),
        distance=8, startangle=np.pi/4):
        '''

        Generates a sequence of 2D (x and z) center-edge-center target triplets, with occasional
        center-edge-far edge catch trials thrown in.

        Parameters
        ----------
        length : int
            The number of target sets in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.
        catchrate: float
            The percent of trials that are far target catch trials. If distance*2

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations
        '''

       # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        theta = []
        for i in range(nblocks):
            temp = np.arange(startangle, startangle+(2*np.pi), 2*np.pi/ntargets)
            np.random.shuffle(temp)
            theta = theta + [temp]
        theta = np.hstack(theta)

        x = distance*np.cos(theta)
        z = distance*np.sin(theta)
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        targs = np.zeros([nblocks*ntargets, 2, 3])
        targs[:,0,0] = x
        targs[:,0,2] = z

        return targs

    @staticmethod
    def rand_target_sequence_3d(length, boundaries=(-18,18,-10,10,-15,15), distance=10):
        '''

        Generates a sequence of 3D target pairs.

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -y, y, -z, z)
        distance : float
            The distance in cm between the targets in a pair.

        Returns
        -------
        pairs : [n x 3 x 2] array of pairs of target locations


        '''

        # Choose a random sequence of points at least "distance" from the edge of
        # the allowed area
        pts = np.random.rand(length, 3)*((boundaries[1]-boundaries[0]-2*distance),
            (boundaries[3]-boundaries[2]-2*distance),
            (boundaries[5]-boundaries[4]-2*distance))
        pts = pts+(boundaries[0]+distance,boundaries[2]+distance,
            boundaries[4]+distance)

        # Choose a random sequence of points on the surface of a sphere of radius
        # "distance"
        theta = np.random.rand(length)*2*np.pi
        phi = np.arccos(2*np.random.rand(length) - 1)
        x = distance*np.cos(theta)*np.sin(phi)
        y = distance*np.sin(theta)*np.sin(phi)
        z = distance*np.cos(theta)

        # Shift points to correct position relative to first sequence and join two
        # sequences together in a trial x coordinate x start/end point array
        pts2 = np.array([x,y,z]).transpose([1,0]) + pts
        pairs = np.array([pts, pts2]).transpose([1,2,0])
        copy = pairs[0:length//2,:,:].copy()

        # Swap start and endpoint for first half of the pairs
        pairs[0:length//2,:,0] = copy[:,:,1]
        pairs[0:length//2,:,1] = copy[:,:,0]

        # Shuffle list of pairs
        np.random.shuffle(pairs)

        return pairs

    @staticmethod
    def rand_pt_to_pt(length=100, boundaries=(-18,18,-12,12), buf=2, seq_len=2):
        '''
        Generates sequences of random postiions in the XZ plane

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
        list
            Each element of the list is an array of shape (seq_len, 3) indicating the target 
            positions to be acquired for the trial.
        '''
        xmin, xmax, zmin, zmax = boundaries
        L = length*seq_len
        pts = np.vstack([np.random.uniform(xmin+buf, xmax-buf, L),
            np.zeros(L), np.random.uniform(zmin+buf, zmax-buf, L)]).T
        targ_seqs = []
        for k in range(length):
            targ_seqs.append(pts[k*seq_len:(k+1)*seq_len])
        return targ_seqs

    @staticmethod
    def rand_target_sequence_2d(length, boundaries=(-18,18,-12,12), distance=10):
        '''

        Generates a sequence of 2D (x and z) target pairs.

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
        pairs : [n x 2 x 2] array of pairs of target locations
        '''

        # Choose a random sequence of points at least "distance" from the boundaries
        pts = np.random.rand(length, 3)*((boundaries[1]-boundaries[0]-2*distance),
            0, (boundaries[3]-boundaries[2]-2*distance))
        pts = pts+(boundaries[0]+distance, 0, boundaries[2]+distance)

        # Choose a random sequence of points on the edge of a circle of radius
        # "distance"
        theta = np.random.rand(length)*2*np.pi
        x = distance*np.cos(theta)
        z = distance*np.sin(theta)

        # Shift points to correct position relative to first sequence and join two
        # sequences together in a trial x coordinate x start/end point array
        pts2 = np.array([x,np.zeros(length),z]).transpose([1,0]) + pts
        pairs = np.array([pts, pts2]).transpose([1,2,0])
        copy = pairs[0:length//2,:,:].copy()

        # Swap start and endpoint for first half of the pairs
        pairs[0:length//2,:,0] = copy[:,:,1]
        pairs[0:length//2,:,1] = copy[:,:,0]

        # Shuffle list of pairs
        np.random.shuffle(pairs)

        return pairs

    @staticmethod
    def rand_target_sequence_2d_centerout(length, boundaries=(-18,18,-12,12),
        distance=10):
        '''

        Generates a sequence of 2D (x and z) target pairs with the first target
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
        pairs : [n x 2 x 2] array of pairs of target locations


        '''

        # Create list of origin targets
        pts1 = np.zeros([length,3])

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        theta = np.random.rand(length)*2*np.pi
        x = distance*np.cos(theta)
        z = distance*np.sin(theta)
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        pts2 = np.array([x,np.zeros(length),z]).transpose([1,0])
        pairs = np.array([pts1, pts2]).transpose([1,2,0])
        
        return pairs

    @staticmethod
    def rand_target_sequence_2d_partial_centerout(length, boundaries=(-18,18,-12,12),distance=10,perc_z=20):
        '''
        PK
        Generates a sequence of 2D (x and z) target pairs with the first target
        always at the origin.

        Parameters
        ----------
        perc_z : float
            The percentage of the z axis to be used for targets 
        length : int
            The number of target pairs in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between the targets in a pair.
        

        Returns
        -------
        pairs : [n x 2 x 2] array of pairs of target locations


        '''
        # Need to get perc_z from settable traits
        # perc_z = traits.Float(0.1, desc="Percent of Y axis that targets move along")
        perc_z=float(10)
        # Create list of origin targets
        pts1 = np.zeros([length,3])

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance"
        
        #Added PK -- to confine z value according to entered boundaries: 
        theta_max = math.asin(boundaries[3]/distance*(perc_z)/float(100))
        theta = (np.random.rand(length)-0.5)*2*theta_max
        
        #theta = np.random.rand(length)*2*np.pi
        
        x = distance*np.cos(theta)*(np.ones(length)*-1)**np.random.randint(1,3,length)
        z = distance*np.sin(theta)
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        pts2 = np.array([x,np.zeros(length),z]).transpose([1,0])

        pairs = np.array([pts1, pts2]).transpose([1,2,0])
        
        return pairs

    @staticmethod
    def rand_multi_sequence_2d_centerout2step(length, boundaries=(-20,20,-12,12), distance=10):
        '''

        Generates a sequence of 2D (x and z) center-edge-far edge target triplets.

        Parameters
        ----------
        length : int
            The number of target pairs in the sequence.
        boundaries: 6 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)
        distance : float
            The distance in cm between consecutive targets.

        Returns
        -------
        targs : [length x 3 x 3] array of pairs of target locations


        '''

        # Create list of origin targets
        pts1 = np.zeros([length,3])

        # Choose a random sequence of points on the edge of a circle of radius 
        # "distance", and matching set with radius distance*2
        theta = np.random.rand(length*10)*2*np.pi
        x1 = distance*np.cos(theta)
        z1 = distance*np.sin(theta)
        x2 = distance*2*np.cos(theta)
        z2 = distance*2*np.sin(theta)

        mask = np.logical_and(np.logical_and(x2>=boundaries[0],x2<=boundaries[1]),np.logical_and(z2>=boundaries[2],z2<=boundaries[3]))
        
        # Join start and end points together in a [trial x coordinate x start/end]
        # array (fill in zeros for endpoint y values)
        pts2 = np.array([x1[mask][:length], np.zeros(length), z1[mask][:length]]).transpose([1,0])
        pts3 = np.array([x2[mask][:length], np.zeros(length), z2[mask][:length]]).transpose([1,0])
        targs = np.array([pts1, pts2, pts3]).transpose([1,2,0])
        
        return targs

class JoystickMulti(ManualControlMulti):

    # #Settable Traits
    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    random_rewards = traits.Float(0,desc="Add randomness to reward, 1: yes, 0: no")
    joystick_speed = traits.Float(20, desc="Radius of cursor")

    is_bmi_seed = True

    def __init__(self, *args, **kwargs):
        super(JoystickMulti, self).__init__(*args, **kwargs)
        self.current_pt=np.zeros([3]) #keep track of current pt
        self.last_pt=np.zeros([3]) #keep track of last pt to calc. velocity
        #self.plant_visible = False
        #self.plant.cursor_color = (0., 1., 0., 0.5)

    def update_report_stats(self):
        super(JoystickMulti, self).update_report_stats()
        start_time = self.state_log[0][1]
        rewardtimes=np.array([state[1] for state in self.state_log if state[0]=='reward'])
        if len(rewardtimes):
            rt = rewardtimes[-1]-start_time
        else:
            rt= np.float64("0.0")

        sec = str(np.int(np.mod(rt,60)))
        if len(sec) < 2:
            sec = '0'+sec
        self.reportstats['Time Of Last Reward'] = str(np.int(np.floor(rt/60))) + ':' + sec
    
    def _test_trial_complete(self, ts):
        if self.target_index==self.chain_length-1 :
            if self.random_rewards:
                if not self.rand_reward_set_flag: #reward time has not been set for this iteration
                    self.reward_time = np.max([2*(np.random.rand()-0.5) + self.reward_time_base, self.reward_time_base/2]) #set randomly with min of base / 2
                    self.rand_reward_set_flag =1;
                    #print self.reward_time, self.rand_reward_set_flag
            return self.target_index==self.chain_length-1
        
    def _test_reward_end(self, ts):
        #When finished reward, reset flag. 
        if self.random_rewards:
            if ts > self.reward_time:
                self.rand_reward_set_flag = 0;
                #print self.reward_time, self.rand_reward_set_flag, ts
        return ts > self.reward_time

    def move_effector(self):
        ''' Returns the 3D coordinates of the cursor. For manual control, uses
        motiontracker data. If no motiontracker data available, returns None'''

        #get data from phidget
        pt = self.joystick.get()
        #print pt

        if len(pt) > 0:

            pt = pt[-1][0]
            x = pt[1]
            y = 1-pt[0]


            pt[0]=1-pt[0]; #Switch L / R axes
            calib = [0.5,0.5] #Sometimes zero point is subject to drift this is the value of the incoming joystick when at 'rest' 
            # calib = [ 0.487,  0.   ]

            #if self.joystick_method==0:                
            if self.joystick_method==0:
                pos = np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                pos[0] = pos[0]*36
                pos[2] = pos[2]*24
                self.current_pt = pos

            elif self.joystick_method==1:
                #vel=np.array([(pt[0]-calib[0]), 0, calib[1]-pt[1]])
                vel = np.array([x-calib[0], 0., y-calib[1]])
                epsilon = 2*(10**-2) #Define epsilon to stabilize cursor movement
                if sum((vel)**2) > epsilon:
                    self.current_pt=self.last_pt+20*vel*(1/60) #60 Hz update rate, dt = 1/60
                else:
                    self.current_pt = self.last_pt

                #self.current_pt = self.current_pt + (np.array([np.random.rand()-0.5, 0., np.random.rand()-0.5])*self.joystick_speed)
                
                if self.current_pt[0] < -25: self.current_pt[0] = -25
                if self.current_pt[0] > 25: self.current_pt[0] = 25
                if self.current_pt[-1] < -14: self.current_pt[-1] = -14
                if self.current_pt[-1] > 14: self.current_pt[-1] = 14

            self.plant.set_endpoint_pos(self.current_pt)
            self.last_pt = self.current_pt.copy()

    @classmethod
    def get_desc(cls, params, report):
        duration = report[-1][-1] - report[0][-1]
        reward_count = 0
        for item in report:
            if item[0] == "reward":
                reward_count += 1
        return "{} rewarded trials in {} min".format(reward_count, duration)

class JoystickMulti2DWindow(JoystickMulti, WindowDispl2D):
    fps = 20.
    def __init__(self,*args, **kwargs):
        super(JoystickMulti2DWindow, self).__init__(*args, **kwargs)

    def _start_wait(self):
        self.wait_time = 0.
        super(JoystickMulti2DWindow, self)._start_wait()
        
    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

