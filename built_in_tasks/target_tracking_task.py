'''
A generic target tracking task
'''
from __future__ import barry_as_FLUFL
from multiprocessing.connection import wait
import numpy as np
import time
import os
import math
import traceback
import random
from collections import OrderedDict

from riglib.experiment import traits, Sequence, FSMTable, StateTransitions
from riglib.stereo_opengl import ik
from riglib import plants

from riglib.stereo_opengl.window import Window
from robot import trajectory
from .target_graphics import *

import matplotlib.pyplot as plt

## Plants
# List of possible "plants" that a subject could control either during manual or brain control
cursor = plants.CursorPlant()
shoulder_anchor = np.array([2., 0., -15])
chain_kwargs = dict(link_radii=.6, joint_radii=0.6, joint_colors=(181/256., 116/256., 96/256., 1), link_colors=(181/256., 116/256., 96/256., 1))
chain_20_20_endpt = plants.EndptControlled2LArm(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)
chain_20_20 = plants.RobotArmGen2D(link_lengths=[20, 20], base_loc=shoulder_anchor, **chain_kwargs)

plantlist = dict(
    cursor=cursor,
    chain_20_20=chain_20_20,
    chain_20_20_endpt=chain_20_20_endpt)

class TargetTracking(Sequence):
    '''
    This is a generic cued target tracking skeleton, to form as a common ancestor to the most
    common type of motor control tracking tasks.
    '''
    status = dict(
        wait = dict(start_trial="trajectory", start_pause="pause"),
        wait_retry = dict(start_trial="trajectory", start_pause="pause"),
        trajectory = dict(enter_target="hold", timeout="timeout_penalty", start_pause="pause"),
        hold = dict(hold_complete="tracking_in", leave_target="hold_penalty", start_pause="pause"),
        tracking_in = dict(trial_complete="reward", leave_target="tracking_out", start_pause="pause"),
        tracking_out = dict(trial_complete="reward", enter_target="tracking_in", tracking_out_timeout="tracking_out_penalty", start_pause="pause"),
        timeout_penalty = dict(timeout_penalty_end="wait", start_pause="pause", end_state=True),
        hold_penalty = dict(hold_penalty_end="wait", hold_penalty_end_retry="wait_retry", start_pause="pause", end_state=True),
        tracking_out_penalty = dict(tracking_out_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True)
        # all end_states will result in trial counter +1, so if you start pause during a penalty state, 
        # the next trial after unpausing will be current trial +2

    )

    # initial state
    state = "wait"
    tries = 0 # Helper variable to keep track of the number of failed attempts at initiating a given trial

    wait_time = traits.Float(1., desc="Length of time in wait state (inter-trial interval)")
    reward_time = traits.Float(.5, desc="Length of reward dispensation")
    timeout_time = traits.Float(10, desc="Time allowed to go between trajectories")
    timeout_penalty_time = traits.Float(1, desc="Length of penalty time for initiation timeout error")
    hold_time = traits.Float(.1, desc="Time of hold required at target before trajectory begins")
    hold_penalty_time = traits.Float(1, desc="Length of penalty time for target hold error")
    tracking_out_time = traits.Float(2.5, desc="Time allowed to be tracking outside the target") # AKA tolerance time
    tracking_out_penalty_time = traits.Float(1, desc="Length of penalty time for tracking out error")
    max_hold_attempts = traits.Int(5, desc='Number of attempts to initiate a trial before skipping to the next one')
    tracking_reward_time = traits.Float(.15, desc="Length of reward for tracking in")
    tracking_reward_interval = traits.Float(.5, desc="Length of time between reward for tracking in (must be greater than reward time)")
    
    def init(self):
        self.trial_dtype = np.dtype([('trial', 'u4'), ('index', 'u4'), ('target', 'f8',(3,)), ('disturbance', 'f8',(3,)), ('is_disturbance', '?')])
        super().init()

        self.frame_index = 0 # index in the frame in a trial
        self.total_distance_error = 0 # Euclidian distance between cursor and target during each trial
        self.trial_timed_out = True # check if the trial is finished
        self.plant_position = []
        self.disturbance_trial = False
        self.repeat_freq_set = False
        self.gen_index = -1

        if self.velocity_control:
            print('VELOCITY CONTROL')
        else:
            print('POSITION CONTROL') # default is position control - see manualcontrolmixin
        self.pos_offset = [0,0,0]
        self.vel_offset = [0,0,0]

    def _parse_next_trial(self):
        '''Get the required data from the generator'''
        # yield idx, pts, disturbance, dis_trajectory, sample_rate :
        self.gen_index, self.targs, self.disturbance_trial, self.disturbance_path, self.sample_rate = self.next_trial # targs and disturbance are same length

        self.targs = np.squeeze(self.targs,axis=0)
        self.disturbance_path = np.squeeze(self.disturbance_path)

        WIDTH, HEIGHT = self.window_size[0], self.window_size[1]
        lookahead = np.zeros((self.lookahead,np.shape(self.targs)[1])) # (30,3)

        self.targs = self.trajectory_amplitude*self.targs
        self.disturbance_path = self.disturbance_amplitude*self.disturbance_path
        # print(np.amax(self.targs), np.amax(self.disturbance_path))

        self.targs = np.concatenate((lookahead, self.targs),axis=0) # (time_length*sample_rate+30,3) # targs and disturbance are no longer same length
    
    def tracking_task_start_wait(self):
        print(self.gen_index)

        self.trial_record['trial'] = self.calc_trial_num()
        self.trial_record['index'] = self.gen_index
        self.trial_record['is_disturbance'] = self.disturbance_trial
        for i in range(len(self.disturbance_path)):
            # Update the data sinks with trial information --> bmi3d_trials
            self.trial_record['target'] = self.targs[i+self.lookahead]
            self.trial_record['disturbance'] = self.disturbance_path[i]
            self.sinks.send("trials", self.trial_record)

        # trial is not finished
        self.trial_timed_out = False

        # number of times this trajectory has been attempted
        self.tries = 0

        # index into trajectory
        self.frame_index = -1

        # number of frames in trajectory
        '''Nothing generic to do.'''
        self.trajectory_length = len(self.targs)

        # saved plant poitions
        self.plant_position = []

    def _start_wait(self):
        # Call parent method to draw the next target capture sequence from the generator
        super()._start_wait()
        if self.sample_rate != self.fps:
            print('WARNING: generator sample rate should equal FSM fps!')

        if self.repeat_freq_set:
            try:
                self.next_trial = next(self.gen)
                self._parse_next_trial()
            except StopIteration:
                self.end_task()

        self.tracking_task_start_wait()

    def _while_wait(self):
        '''Nothing generic to do.'''
        pass

    def _start_wait_retry(self):
        self.tracking_task_start_wait()

    def _while_wait_retry(self):
        '''Nothing generic to do.'''
        pass

    def _start_trajectory(self):
        self.tries += 1
        self.frame_index += 1

    def _while_trajectory(self):
        '''Nothing generic to do.'''
        pass        

    def _end_trajectory(self):
        '''Nothing generic to do.'''
        pass

    def _start_hold(self):
        '''Nothing generic to do.'''
        pass

    def _while_hold(self):
        self.pos_offset = [0,0,0]
        self.vel_offset = [0,0,0]

    def _end_hold(self):
        '''Nothing generic to do.'''
        pass

    def _start_tracking_in(self):
        '''Nothing generic to do.'''
        pass

    def _while_tracking_in(self):
        '''Nothing generic to do.'''
        pass

    def _end_tracking_in(self):
        '''Nothing generic to do.'''
        pass

    def _start_tracking_out(self):
        '''Nothing generic to do.'''
        pass

    def _while_tracking_out(self):
        '''Nothing generic to do.'''
        pass

    def _end_tracking_out(self):
        '''Nothing generic to do.'''
        pass

    def _start_timeout_penalty(self):
        '''Nothing generic to do.'''
        # self._increment_tries()
        pass

    def _while_timeout_penalty(self):
        self.pos_offset = [0,0,0]
        self.vel_offset = [0,0,0]

    def _end_timeout_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _start_hold_penalty(self):
        '''Nothing generic to do.'''
        # self._increment_tries()
        pass

    def _while_hold_penalty(self):
        self.pos_offset = [0,0,0]
        self.vel_offset = [0,0,0]

    def _end_hold_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _start_tracking_out_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _while_tracking_out_penalty(self):
        self.pos_offset = [0,0,0]
        self.vel_offset = [0,0,0]

    def _end_tracking_out_penalty(self):
        '''Nothing generic to do.'''
        pass

    def _start_reward(self):
        '''Nothing generic to do.'''
        pass

    def _while_reward(self):
        self.pos_offset = [0,0,0]
        self.vel_offset = [0,0,0]

    def _end_reward(self):
        '''Nothing generic to do.'''
        pass

    def _start_pause(self):
        '''Nothing generic to do.'''
        pass

    def _while_pause(self):
        '''Nothing generic to do.'''
        self.pos_offset = [0,0,0]
        self.vel_offset = [0,0,0]

    def _end_pause(self):
        '''Nothing generic to do.'''
        pass

    ################## State transition test functions ##################
    def _test_start_trial(self, time_in_state):
        '''Start next trial automatically. You may want this to instead be
            - a random delay
            - require some initiation action
        '''
        return time_in_state > self.wait_time

    def _test_timeout(self, time_in_state):
        return time_in_state > self.timeout_time

    def _test_hold_complete(self, time_in_state):
        '''
        Test whether the target is held long enough to declare the
        trial a success

        Possible options
            - Target held for the minimum requred time (implemented here)
            - Sensorized object moved by a certain amount
            - Sensorized object moved to the required location
            - Manually triggered by experimenter
        '''
        return time_in_state > self.hold_time

    def _test_trial_complete(self, time_in_state):
        '''Test whether the trajectory is finished'''
        return self.frame_index + self.lookahead == self.trajectory_length

    def _test_tracking_out_timeout(self, time_in_state):
        return time_in_state > self.tracking_out_time

    def _test_timeout_penalty_end(self, time_in_state):
        return time_in_state > self.timeout_penalty_time #or self.pause

    def _test_hold_penalty_end(self, time_in_state):
        return (time_in_state > self.hold_penalty_time) and (self.tries==self.max_hold_attempts) #or self.pause

    def _test_hold_penalty_end_retry(self, time_in_state):
        return (time_in_state > self.hold_penalty_time) and (self.tries<self.max_hold_attempts) #or self.pause

    def _test_tracking_out_penalty_end(self, time_in_state):
        return time_in_state > self.tracking_out_penalty_time #or self.pause

    def _test_reward_end(self, time_in_state):
        return time_in_state > self.reward_time

    def _test_enter_target(self, time_in_state):
        '''This function is task-specific and not much can be done generically'''
        return False

    def _test_leave_target(self, time_in_state):
        '''This function is task-specific and not much can be done generically'''
        return False

    def _test_start_pause(self, time_in_state):
        return self.pause

    def _test_end_pause(self, time_in_state):
        return not self.pause
    
    def update_report_stats(self):
        '''
        see experiment.Experiment.update_report_stats for docs
        '''
        super().update_report_stats()
        self.reportstats['Trial #'] = self.calc_trial_num()
        self.reportstats['Reward/min'] = np.round(self.calc_events_per_min('reward', 120.), decimals=2)
        

class ScreenTargetTracking(TargetTracking, Window):
    """Concrete implementation of Target Tracking task where the target is moving and
    are tracked by holding the cursor within the moving target"""

    limit2d = traits.Bool(True, desc="Limit cursor movement to 2D")
    limit1d = traits.Bool(True, desc="Limit cursor movement to 1D")

    sequence_generators = [
        'tracking_target_chain', 'tracking_target_debug', 'tracking_target_training'
    ]

    hidden_traits = ['cursor_color', 'trajectory_color', 'cursor_bounds', 'cursor_radius', 'plant_hide_rate', 'starting_pos']
    targets = []

    is_bmi_seed = True

    # Runtime settable traits
    target_radius = traits.Float(2, desc="Radius of targets in cm") #2,0.75
    trajectory_radius = traits.Float(.5, desc="Radius of targets in cm")
    trajectory_color = traits.OptionsList("gold", *target_colors, desc="Color of the trajectory", bmi3d_input_options=list(target_colors.keys()))
    target_color = traits.OptionsList("yellow", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))
    plant_hide_rate = traits.Float(0.0, desc='If the plant is visible, specifies a percentage of trials where it will be hidden')
    plant_type = traits.OptionsList(*plantlist, bmi3d_input_options=list(plantlist.keys()))
    plant_visible = traits.Bool(True, desc='Specifies whether entire plant is displayed or just endpoint')
    cursor_radius = traits.Float(.5, desc='Radius of cursor in cm')
    cursor_color = traits.OptionsList("pink", *target_colors, desc='Color of cursor endpoint', bmi3d_input_options=list(target_colors.keys()))
    cursor_bounds = traits.Tuple((-10., 10., 0., 0., -10., 10.), desc='(x min, x max, z min, z max, y min, y max)')
    starting_pos = traits.Tuple((5., 0., 5.), desc='Where to initialize the cursor')
    fps = traits.Float(60, desc="Rate at which the FSM is called in Hz") # originally set by class Experiment
    trajectory_amplitude = traits.Float(1, desc='Scale factor applied to the trajectory')
    disturbance_amplitude = traits.Float(1, desc='Scale factors applied to the disturbance')
    always_1d = traits.Bool(False, desc='always constrain movement to 1d')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the plant
        if not hasattr(self, 'plant'):
            self.plant = plantlist[self.plant_type]
        self.plant.set_bounds(np.array(self.cursor_bounds))
        self.plant.set_color(target_colors[self.cursor_color])
        self.plant.set_cursor_radius(self.cursor_radius)
        self.plant_vis_prev = True
        self.cursor_vis_prev = True
        self.lookahead = 30 # number of frames to create a "lookahead" window of 0.5 seconds (half the screen)
        self.original_limit1d = self.limit1d # keep track of original settable trait
        
        if not self.always_1d:
            self.limit1d = False # allow 2d movement before center-hold initiation
        
        # Add graphics models for the plant and targets to the window
        if hasattr(self.plant, 'graphics_models'):
            for model in self.plant.graphics_models:
                self.add_model(model)

        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)
        if instantiate_targets:
            # This is the center target being followed by the user
            self.target = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])

            # This is the trajectory that spans the screen
            self.trajectory = VirtualCableTarget(target_radius=self.trajectory_radius, target_color=target_colors[self.trajectory_color])

            # This is the progress bar
            self.bar = VirtualRectangularTarget(target_width=1, target_height=0, target_color=(0., 1., 0., 0.75), starting_pos=[0,0,9])
            # print('INIT TRAJ')

        # Declare any plant attributes which must be saved to the HDF file at the _cycle rate
        for attr in self.plant.hdf_attrs:
            self.add_dtype(*attr)

    def init(self):
        self.add_dtype('trial', 'u4', (1,))
        self.add_dtype('gen_idx', 'int', (1,)) # dtype needs to be able to represent -1
        self.add_dtype('plant_visible', '?', (1,))
        self.add_dtype('current_target', 'f8', (3,))
        self.add_dtype('current_disturbance', 'f8', (3,)) # see task_data['manual_input'] for cursor position without added disturbance
        self.add_dtype('current_target_validate', 'f8', (3,))
        super().init()
        self.plant.set_endpoint_pos(np.array(self.starting_pos))

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen
        '''
        self.move_effector(pos_offset=np.asarray(self.pos_offset), vel_offset=np.asarray(self.vel_offset))

        # Run graphics commands to show/hide the plant if the visibility has changed
        self.update_plant_visibility()
        self.task_data['plant_visible'] = self.plant_visible

        # Save plant status to HDF file
        plant_data = self.plant.get_data_to_save() # for cursor plant, save cursor position
        for key in plant_data:
            self.task_data[key] = plant_data[key] # task_data['cursor']

        # Update the trial index
        self.task_data['trial'] = self.calc_trial_num()
        self.task_data['gen_idx'] = self.gen_index
        # print(self.task_data['gen_idx'])
        
        # Save the target position at each cycle. 
        if self.trial_timed_out:
             self.task_data['current_target'] = [np.nan,np.nan,np.nan]
             self.task_data['current_disturbance'] = [np.nan,np.nan,np.nan]
             self.task_data['current_target_validate'] = self.target.get_position() # default VirtualCircularTarget position is [0,0,0]
        else:
            self.task_data['current_target'] = self.targs[self.frame_index+self.lookahead]
            self.task_data['current_disturbance'] = self.disturbance_path[self.frame_index]
            self.task_data['current_target_validate'] = self.target.get_position()

        super()._cycle()

    def move_effector(self):
        '''Move the end effector, if a robot or similar is being controlled'''
        pass

    def run(self):
        '''
        See experiment.Experiment.run for trajectorydocumentation.
        '''
        # Fire up the plant. For virtual/simulation plants, this does little/nothing.
        self.plant.start()
        
        # Include some cleanup in case the parent class has errors
        try:
            super().run()
        finally:
            self.plant.stop()

    ##### HELPER AND UPDATE FUNCTIONS ####
    def update_plant_visibility(self):
        ''' Update plant visibility'''
        if self.plant_visible != self.plant_vis_prev:
            self.plant_vis_prev = self.plant_visible
            self.plant.set_visibility(self.plant_visible)

    def update_frame(self):
        self.target.move_to_position(np.array([0,0,self.targs[self.frame_index+self.lookahead][2]])) # xzy
        self.trajectory.move_to_position(np.array([-self.frame_index/3,0,0])) # same update constant works for 60 and 120 hz
        self.target.show()
        self.trajectory.show()
        self.frame_index +=1

    #### TEST FUNCTIONS ####
    def _test_enter_target(self, time_in_state):
        '''
        Test if the cursor is inside the center target 
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target.get_position())
        return d <= (self.target_radius - self.cursor_radius) or super()._test_enter_target(time_in_state) or (self.hold_time==0 and self.state=='trajectory')

    def _test_leave_target(self, time_in_state):
        '''
        Test if the cursor is outside the center target
        '''
        cursor_pos = self.plant.get_endpoint_pos()
        d = np.linalg.norm(cursor_pos - self.target.get_position())
        return d > (self.target_radius - self.cursor_radius) or super()._test_leave_target(time_in_state)

    #### STATE FUNCTIONS ####
    def setup_start_wait(self):
        if self.calc_trial_num() == 0:
            # Instantiate the targets here so they don't show up in any states that might come before "wait" 
            for model in self.target.graphics_models:
                self.add_model(model)
                self.target.hide()
                
            for model in self.trajectory.graphics_models:
                self.add_model(model)
                self.trajectory.hide()

            for model in self.bar.graphics_models:
                self.add_model(model)
                self.bar.hide()

        # Allow 2d movement
        if not self.always_1d:
            self.limit1d = False

        # Set up for progress bar
        self.bar_width = 12        
        self.tracking_frame_index = 0
        
        # Set up the next trajectory
        next_trajectory = np.array(np.squeeze(self.targs)[:,2])
        next_trajectory[:self.lookahead] = next_trajectory[self.lookahead]

        if hasattr(self, 'trajectory'):
            for model in self.trajectory.graphics_models:
                self.remove_model(model)
            del self.trajectory

        self.trajectory = VirtualCableTarget(target_radius=self.trajectory_radius, target_color=target_colors[self.trajectory_color], trajectory=next_trajectory)

        for model in self.trajectory.graphics_models:
                self.add_model(model)

        self.target.hide()
        self.trajectory.hide()
        # self.in_end_state = False      

    def setup_screen_reset(self):
        # Hide target and trajectory
        self.target.hide()
        self.target.reset()
        self.trajectory.hide()
        self.trajectory.reset()
        self.bar.hide()
        self.bar.reset()

    def _start_wait(self):
        super()._start_wait()
        self.setup_start_wait()
        print('WAIT')

    def _while_wait(self):
        super()._while_wait()

    def _start_wait_retry(self):
        super()._start_wait_retry()
        self.setup_start_wait()
        # print('WAIT RETRY')

    def _while_wait_retry(self):
        super()._while_wait_retry()
        # # Add disturbance

    def _start_trajectory(self):
        super()._start_trajectory()
        if self.frame_index == 0:
            self.target.move_to_position(np.array([0,0,self.targs[self.frame_index+self.lookahead][2]])) # tablet screen x-axis ranges -19,19, center 0
            self.trajectory.move_to_position(np.array([0,0,0])) # tablet screen x-axis ranges 0,41.33333, center 22ish
            # print(self.target.get_position())
            # print(self.trajectory.get_position())

            self.target.show()
            self.trajectory.show()
            # print('SHOW TRAJ')
            self.sync_event('TARGET_ON')

    def _while_trajectory(self):
        super()._while_trajectory()
        # # Add disturbance

    def _start_hold(self):
        super()._start_hold()
        # print('START HOLD')
        self.sync_event('TRIAL_START')
        # Cue successful tracking
        self.target.cue_trial_end_success()

    def _while_hold(self):
        super()._while_hold()
        # # Add disturbance

    def _start_tracking_in(self):
        super()._start_tracking_in()
        # print('START TRACKING')
        self.sync_event('CURSOR_ENTER_TARGET')
        # Revert to settable trait
        self.limit1d = self.original_limit1d
        # Cue successful tracking
        self.target.cue_trial_end_success()

    def _while_tracking_in(self):
        super()._while_tracking_in()
    
        # Add disturbance
        cursor_pos = self.plant.get_endpoint_pos()
        if self.disturbance_trial == True:
            if self.velocity_control:
                # TODO check manualcontrolmixin for how to implement velocity control
                self.vel_offset = (cursor_pos + self.disturbance_path[self.frame_index])*1/self.fps
            else: 
                # position control
                self.pos_offset = self.disturbance_path[self.frame_index]

        # Move target and trajectory to next frame so it appears to be moving
        self.update_frame()

        # Check if the trial is over and there are no more target frames to display
        if self.frame_index+self.lookahead >= np.shape(self.targs)[0]:
            self.trial_timed_out = True
            

    def _start_tracking_out(self):
        super()._start_tracking_out()
        # print('STOP TRACKING')
        self.sync_event('CURSOR_LEAVE_TARGET')
        # Reset target color
        self.target.reset()

    def _while_tracking_out(self):
        super()._while_tracking_out()

        # Add disturbance
        cursor_pos = self.plant.get_endpoint_pos()
        if self.disturbance_trial == True:
            if self.velocity_control:
                # TODO check manualcontrolmixin for how to implement velocity control
                self.vel_offset = (cursor_pos + self.disturbance_path[self.frame_index])*1/self.fps
            else: 
                # position control
                self.pos_offset = self.disturbance_path[self.frame_index]

        # Move target and trajectory to next frame so it appears to be moving
        self.update_frame()

        # Check if the trial is over and there are no more target frames to display
        if self.frame_index+self.lookahead >= np.shape(self.targs)[0]:
            self.trial_timed_out = True

    def _start_timeout_penalty(self):
        super()._start_timeout_penalty()
        # print('START TIMEOUT')
        self.sync_event('TIMEOUT_PENALTY')

        # self.in_end_state = True
        self.setup_screen_reset()

        # skip to next generated trial using same freq set
        self.repeat_freq_set = True

    def _while_timeout_penalty(self):
        super()._while_timeout_penalty()
        # # Add disturbance

    def _end_timeout_penalty(self):
        super()._end_timeout_penalty()
        self.sync_event('TRIAL_END')
            
    def _start_hold_penalty(self):
        super()._start_hold_penalty()
        # print('START HOLD TIMEOUT')
        self.sync_event('HOLD_PENALTY')

        # self.in_end_state = True
        self.setup_screen_reset()

        # skip to next generated trial using same freq set
        self.repeat_freq_set = True

    def _while_hold_penalty(self):
        super()._while_hold_penalty()
        # # Add disturbance

    def _end_hold_penalty(self):
        super()._end_hold_penalty()
        # print('TRIAL END')
        self.sync_event('TRIAL_END')

    def _start_tracking_out_penalty(self):
        super()._start_tracking_out_penalty()
        # print('START TRACKING TIMEOUT')
        self.sync_event('OTHER_PENALTY')
        # self.in_end_state = True
        # Cue failed trial
        self.target.cue_trial_end_failure()

        # skip to next generated trial using same freq set
        self.repeat_freq_set = True

    def _while_tracking_out_penalty(self):
        super()._while_tracking_out_penalty()
        # # Add disturbance

    def _end_tracking_out_penalty(self):
        super()._end_tracking_out_penalty()
        # print('TRIAL END')
        self.sync_event('TRIAL_END')
        self.setup_screen_reset()

    def _start_reward(self):
        super()._start_reward()
        # print('REWARD')
        self.sync_event('REWARD')
        # self.in_end_state = True
        # Cue successful trial
        self.target.cue_trial_end_success()
        self.reward_frame_index = 0

        # use next generated trial using other freq set
        self.repeat_freq_set = False

    def _while_reward(self):
        super()._while_reward()
        # # Add disturbance

    def _end_reward(self):
        super()._end_reward()
        # print('TRIAL END')
        self.sync_event('TRIAL_END')
        self.setup_screen_reset()

    def _start_pause(self):
        super()._start_pause()
        # print('START PAUSE')
        self.sync_event('PAUSE_START') # self.sync_event('PAUSE_START') - this will overwrite TRIAL_END whenever you pause during an end state
        self.setup_screen_reset()

    def _while_pause(self):
        super()._while_pause()

    def _end_pause(self):
        super()._end_pause()
        # print('END PAUSE')
        self.sync_event('PAUSE_END')

    @staticmethod
    def calc_sum_of_sines(times, frequencies, amplitudes, phase_shifts):
        '''
        Generates the trajectories for the experiment
        '''
        t = times
        t = np.asarray(t).copy(); t.shape = (t.size,1)

        f = frequencies
        f = f.copy()
        f = np.reshape(f, (1,f.size))

        a = amplitudes
        a = a.copy()
        a = np.reshape(a, (1,a.size))

        p = phase_shifts        
        p = p.copy()
        p = np.reshape(p, (1,p.size))

        assert f.shape == a.shape == p.shape,"Shape of frequencies, amplitudes, and phase shifts must match"

        o = np.ones(t.shape)
        trajectory = np.sum(np.dot(o,a) * np.sin(2*np.pi*(np.dot(t,f) + np.dot(o,p))),axis=1)
        A = np.sum(np.dot(o,a)[0])
        
        return trajectory, A

    @staticmethod
    def calc_sum_of_sines_ramp(times, ramp, ramp_down, frequencies, amplitudes, phase_shifts):
            '''
            Adds a 1/t ramp up and ramp down at the start and end so the trajectories start and end at zero.
            '''
            t = times
            t = np.asarray(t).copy(); t.shape = (t.size,1)

            r = ramp
            rd = ramp_down

            trajectory, A = ScreenTargetTracking.calc_sum_of_sines(t, frequencies, amplitudes, phase_shifts)

            if r > 0:
                trajectory *= (( t*(t <= r)/r + (t > r) ).flatten())**2
                # trajectory *= (((t*(t <= r)/r) + ((t > r) & (t < (t[-1]-r))) + ((t[-1]-t)*(t >= (t[-1]-r))/r)).flatten())**2

            if rd > 0:
                trajectory *= (( (t < (t[-1]-rd)) + ((t[-1]-t)*(t >= (t[-1]-rd))/rd) ).flatten())**2

            return trajectory, A

    @staticmethod
    def generate_trajectories(num_trials=2, time_length=20, seed=40, sample_rate=120, base_period=20, ramp=0, ramp_down=0, num_primes=8):
        '''
        Sets up variables and uses prime numbers to call the above functions and generate then trajectories
        ramp is time length for preparatory lines
        '''
        np.random.seed(seed)
        hz = sample_rate # Hz -- sampling rate
        dt = 1/hz # sec -- sampling period

        T0 = base_period # sec -- base period
        w0 = 1./T0 # Hz -- base frequency

        r = ramp # "ramp up" duration (see sum_of_sines_ramp)
        rd = ramp_down # "ramp down" duration (see sum_of_sines_ramp)
        P = time_length/T0 # number of periods in signal
        T = P*T0+r+rd # sec -- signal duration
        dw = 1./T # Hz -- frequency resolution
        W = 1./dt/2 # Hz -- signal bandwidth

        full_primes = np.asarray([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 
            101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199])
        primes = full_primes[:num_primes]
        # primes_ind = np.where(full_primes <= T0)
        # primes = full_primes[primes_ind]

        f = primes*w0 # stimulated frequencies
        f_ref = f.copy()
        f_dis = f.copy()

        a = 1/(1+np.arange(f.size)) # amplitude
        a_ref = a.copy()
        a_dis = a.copy()

        o = np.random.rand(num_trials,primes.size) # phase offset
        o_ref = o.copy()
        o_dis = o.copy()*0.8

        t = np.arange(0,T,dt) # time samplesseed
        w = np.arange(0,W,dw) # frequency samples

        N = t.size # = T/dt -- number of samples
        
        # create trials dictionary
        trials = dict(
            id=np.arange(num_trials), times=np.tile(t,(num_trials,1)), ref=np.zeros((num_trials,N)), dis=np.zeros((num_trials,N))
            )

        # randomize order of first two trials to provide random starting point
        order = np.random.choice([0,1])
        if order == 0:
            trial_order = [(1,'E','O'),(1,'O','E')]
        elif order == 1:
            trial_order = [(1,'O','E'),(1,'E','O')]

        # generate reference and disturbance trajectories for all trials
        for trial_id, (num_reps,ref_ind,dis_ind) in enumerate(trial_order*int(num_trials/2)):   
            if ref_ind == 'E': 
                sines_r = np.arange(len(primes))[0::2] # use even indices
            elif ref_ind == 'O': 
                sines_r = np.arange(len(primes))[1::2] # use odd indices
            else:
                sines_r = np.arange(len(primes))
            if dis_ind == 'E':
                sines_d = np.arange(len(primes))[0::2]
            elif dis_ind == 'O':
                sines_d = np.arange(len(primes))[1::2]
            else:
                sines_d = np.arange(len(primes))
            
            ref_trajectory, ref_A = ScreenTargetTracking.calc_sum_of_sines_ramp(t, r, rd, f_ref[sines_r], a_ref[sines_r], o_ref[trial_id][sines_r])
            dis_trajectory, dis_A = ScreenTargetTracking.calc_sum_of_sines_ramp(t, r, rd, f_dis[sines_d], a_dis[sines_d], o_dis[trial_id][sines_d])
            
            # normalized trajectories
            trials['ref'][trial_id] = ref_trajectory/ref_A   # previously, denominator was np.sum(a_ref)
            trials['dis'][trial_id] = dis_trajectory/dis_A   # previously, denominator was np.sum(a_dis)
            # print(trial_order, ref_A, dis_A)
        
        return trials, trial_order

    @staticmethod
    def generate_trajectory(primes, base_period, ramp = .0):
        '''
        Sets up variables and uses prime numbers to call the above functions and generate then trajectories
        ramp is time length for preparatory lines
        '''
        hz = 60 # Hz -- sampling rate
        dt = 1/hz # sec -- sampling period

        T0 = base_period # sec -- base period
        w0 = 1/T0 # Hz -- base frequency

        P = 1 # number of periods in signal
        T = P*T0 # sec -- signal duration
        r = ramp # "ramp" duration (see sum_of_sines_ramp)
        dw = 1/T # Hz -- frequency resolution
        W = dw*T/dt/2 # Hz -- signal bandwidth

        f = primes # stimulated frequencies
        a = 1/(1+np.arange(f.size)) # amplitude
        o = 2*np.pi*np.random.rand(primes.size) # phase offset

        t = np.arange(0,T,dt) # time samples
        w = np.arange(0,W,dw) # frequency samples

        N = t.size # = T/dt -- number of samples

        #trajectory = ScreenTargetTracking.calc_sum_of_sines_ramp(t, r, f, a, o/(2*np.pi))
        trajectory = ScreenTargetTracking.calc_sum_of_sines_ramp(t, r, f, a, o)
        normalized_trajectory = trajectory/np.sum(a)
        return normalized_trajectory
    
    ### Generator functions ####
    @staticmethod
    def tracking_target_chain(nblocks=1, ntrials=2, time_length=20, ramp=0, ramp_down=0, num_primes=8, seed=40, sample_rate=120, disturbance=True, boundaries=(-10,10,-10,10)):
        '''
        Generates a sequence of 1D (z axis) target trajectories

        Parameters
        ----------
        nblocks : int
            The number of blocks in the session
        ntrials : int
            The number of trials in a block
        time_length : int
            The length of one trial in seconds
        seed : int
            The seed for the random generator
        sample_rate : int
            The sample rate of the generated trajectories
        ramp : float
            The length of ramp up into a trial in seconds
        disturbance : boolean
            Whether to add disturbance to the cursor (disturbance is generated regardless)
        boundaries: 4 element tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        idx : [nblocks*ntrials x 1] array of trial indices
        pts : [nblocks*ntrials x 1] array of 3D target coordinates
        disturbance : boolean
        dis_trajectory : [nblocks*ntrials x 1] array of 3D disturbance coordinates
        '''
        idx = 0
        base_period = 20
        for block_id in range(nblocks):                
            trials, trial_order = ScreenTargetTracking.generate_trajectories(
                num_trials=ntrials, time_length=time_length, seed=seed, sample_rate=sample_rate, base_period=base_period, ramp=ramp, ramp_down=ramp_down, num_primes=num_primes
                )
            for trial_id in range(ntrials):
                pts = []
                ref_trajectory = np.zeros((int((time_length+ramp+ramp_down)*sample_rate),3))
                dis_trajectory = ref_trajectory.copy()
                ref_trajectory[:,2] = trials['ref'][trial_id]
                dis_trajectory[:,2] = trials['dis'][trial_id] # scale will determine lower limit of target size for perfect tracking
                pts.append(ref_trajectory)
                yield idx, pts, disturbance, dis_trajectory, sample_rate
                idx += 1

    @staticmethod
    def tracking_target_debug(nblocks=1, ntrials=2, time_length=20, seed=40, sample_rate=60, ramp=0, disturbance=True, boundaries=(-10,10,-10,10)):
        '''
        Generates a sequence of 1D (z axis) target trajectories for debugging
        '''
        idx = 0
        base_period = 20
        for block_id in range(nblocks):                
            trials, trial_order = ScreenTargetTracking.generate_trajectories(
                num_trials=ntrials, time_length=time_length, seed=seed, sample_rate=sample_rate, base_period=base_period, ramp=ramp
                )
            for trial_id in range(ntrials):
                # if trial_id==0:
                #     plt.figure(); plt.plot(trials['times'][trial_id],trials['ref'][trial_id])
                #     plt.plot(trials['times'][trial_id],trials['dis'][trial_id])
                #     plt.show()
                pts = []
                ref_trajectory = np.zeros((int((time_length+ramp)*sample_rate),3))
                dis_trajectory = ref_trajectory.copy()
                ref_trajectory[:,2] = trials['ref'][trial_id]
                dis_trajectory[:,2] = trials['dis'][trial_id] # scale will determine lower limit of target size for perfect tracking
                pts.append(ref_trajectory)
                yield idx, pts, disturbance, dis_trajectory, sample_rate
                idx += 1
    
    @staticmethod
    def tracking_target_training(nblocks=1, ntrials=2, time_length=5, frequencies = [1,.75], boundaries=(-10,10,-10,10)):
        '''
        Generates a sequence of 1D (z axis) target trajectories for training

        Parameters
        ----------
        nblocks : int
            The number of tracking trials in the sequence.
        ntrials : int
            The number trials in a block
        time_length : int
        frequencies: numpy.ndarray
            A list of frequencies used to generate the trajectories
        boundaries: 4 element Tuple
            The limits of the allowed target locations (-x, x, -z, z)

        Returns
        -------
        [nblocks*ntrials x 1] array of tuples containing trial indices and [time_length*60 x 3] target coordinates
        '''
        idx = 0
        buffer_space_bef = int(60*1.3) # 1.3 seconds of straight line before trial #78 frames
        buffer_space_aft = int(60*1.5) # 1.5 seconds of straight line before trial #90 frames
        frames = int(np.round(time_length*60)) #300 frames when time_length=5
        y_primes_freq = np.array(frequencies)

        for i in range(nblocks):
            for j in range(ntrials):
                
                disturbance = False
                disturbance_path = np.zeros((frames+buffer_space_bef+buffer_space_aft,1))
                trajectory = np.zeros((frames+buffer_space_bef+buffer_space_aft,3))
                sum_of_sins_path = ScreenTargetTracking.generate_trajectory(y_primes_freq,time_length)
                pts = []
                trajectory[:,2] = 5*np.concatenate((sum_of_sins_path[0]*np.ones(buffer_space_bef),sum_of_sins_path,sum_of_sins_path[-1]*np.ones(buffer_space_aft)))
                pts.append(trajectory)
                yield idx, pts, disturbance, disturbance_path, None
                idx += 1