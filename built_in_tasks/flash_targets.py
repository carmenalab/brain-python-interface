'''
Tasks which control a plant under pure machine control. Used typically for initializing BMI decoder parameters.
'''
import numpy as np
import os
import tables
import time
import subprocess
import tempfile
from OpenGL.GL import GL_RGB, GL_RGB8, GL_UNSIGNED_BYTE
import pygame

from riglib.experiment import traits, Experiment
from riglib.bmi.state_space_models import StateSpaceEndptVel2D
from riglib.bmi.bmi import Decoder, BMILoop, MachineOnlyFilter
from riglib.bmi.extractor import DummyExtractor
from riglib.stereo_opengl.primitives import TexPlane
from riglib.stereo_opengl.textures import Texture
from riglib.stereo_opengl.window import Window, WindowDispl2D, Window2D
from config.rig_defaults import window as window_defaults

from built_in_tasks.manualcontrolmultitasks import ScreenTargetCapture
from built_in_tasks.bmimultitasks import BMIControlMulti
from built_in_tasks.target_capture_task_eye import ScreenTargetCapture_Saccade

from .target_graphics import *

from .bmimultitasks import BMIControlMultiEyeConstrained

class FlashTargets(ScreenTargetCapture_Saccade):
    '''
    # Flash Targets
    This task comprises of a white rectangle to fixate on to intialize the trial. 
    Then a series of circular peripheral targets are flashed while recquiring the animal to maintain the central fixation.

    ## Key Features:
    - arduino_sync
    - autostart
    - ecube_start
    - eye_calibration
    - eye_streaming
    - online_analysis
    - optitrack
    - reward_system
    - rig1_sync
    - saveHDF
    - test_2D

    ## Key Parameters:
    - hold time: determines fixation time and total trial time
    - target_flash_time_s: Tuple, The amount of time in seconds that the peripheral targets are flashed on (target_flash_time_s[0] and off (target_flash_time_s[1])
    - flash_buffer_time_s: Float, The amount of time after the fixation starts and before the fixation ends where no peripheral targets will be shown
    - task_id_for_eye_calibration: Set to the correct ID of the eye calibration or manual control task

    ## Parameters to watch:
    - keyboard_control
    - show_eye_pos
    - session_length

    
    ## Todo
    1. blink_time_threshold: implement logic into a feature so taht we can use it for multiple tasks
    2. Reorganize the functions to improve readibility
    '''

    sequence_generators = [
        'centerout_2D_chain', 'out_2D', 'out_2D_select', 'centerout_2D', 'centeroutback_2D', 'centerout_2D_select', 'rand_target_chain_2D', 'rand_same_target_chain_2D', 
        'rand_target_chain_3D', 'corners_2D', 'centerout_tabletop', 'out_2D_square', 'centerout_2D_square'
    ]
    blink_time_threshold = traits.Float(0.1, desc="The amount of time in seconds that " \
    "the eyes can be closed before triggering a fixation break, measured by eye_diam=0")
    
    peripheral_target_radius = traits.Float(2, desc="Radius of targets in cm")
    peripheral_target_color = traits.OptionsList("yellow", *target_colors, desc="Color of the target", bmi3d_input_options=list(target_colors.keys()))

    #This is the variable that sets the total flash on and flash off time
    target_flash_time_s = traits.Tuple((0.2, 0.2), desc="The amount of time in seconds that " \
    "the peripheral targets are flashed on (target_flash_time_s[0] and off (target_flash_time_s[1])")

    #this is the variable that sets the buffer windown at the beginning and end of the hold period where no peripheral target will be shown
    flash_buffer_time_s = traits.Float(0.2, desc='The amount of time after the fixation starts and before the fixation ends where no ' \
    'peripheral targets will be shown')
    
    hold_time = traits.Float(0.8, desc="Length of central hold, ie the central fixation during the entire task. It must include the beginning and end buffer and a integer number of flash on/off cycles. Given default buffer and flash times, valid hold times are 0.8 (1 flash), 1.2 (2 flashes), 1.6 etc...")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.most_recent_open_eye = 0
        #Here we calculate the total number of flash targets that can be shown in a single trial
        self.total_flashes = np.round((self.hold_time - 2*(self.flash_buffer_time_s))/(self.target_flash_time_s[0] + self.target_flash_time_s[1]), decimals=3)
        assert self.total_flashes==np.floor(self.total_flashes), f"The hold time needs to be appropriately set to allow for the start/end buffer and a integer number of flashes. Current hold time should be modified to {np.floor(self.total_flashes)*(self.target_flash_time_s[0] + self.target_flash_time_s[1]) + 2*(self.flash_buffer_time_s)}"
        assert self.total_flashes>0, f"Hold time {self.hold_time} is too short to show any peripheral targets with the current flash buffer time {self.flash_buffer_time_s} and flash on/off times {self.target_flash_time_s}"

        #Flag to track the number of flashes shown per trial
        self.flash_tracker = 0
        # Instantiate the targets
        instantiate_targets = kwargs.pop('instantiate_targets', True)

        if instantiate_targets:

            # Control transparency of targets
            new_color1 = list(target_colors[self.target_color])
            new_color1[3] = self.init_eye_target_alpha
            new_color2 = list(target_colors[self.peripheral_target_color])

            # 2 targets for delay
            #Target 1 is the central fixation target we are using as the eye target
            target1 = VirtualRectangularTarget(target_width=self.target_radius, target_height=self.target_radius/2, target_color=new_color1)
            #Target 2 will be the peripheral 'flash' target; needs to be modified to regular center out target
            target2 = VirtualCircularTarget(target_radius=self.peripheral_target_radius, target_color=new_color2)

            self.targets = [target1, target2]

            self.offset_cube = np.array([0,0,self.target_radius/2]) # To center the cube target
            self.center_target_index = 0
            self.center_target_position = np.array([0,0,0] )

    
    

    status = dict(
        wait = dict(start_trial="target", start_pause="pause"),
        target = dict(enter_target="hold_start_buffer", start_pause="pause"),
        hold_start_buffer = dict(leave_target="hold_penalty", start_flash_cycle='hold_flash_on', start_pause="pause"),
        hold_flash_on = dict(leave_target="hold_penalty", flash_complete="hold_flash_off", start_pause="pause"),
        hold_flash_off = dict(leave_target="hold_penalty", flash_interval_complete = "hold_flash_on", flash_cycle_complete='hold_end_buffer', start_pause="pause"),
        hold_end_buffer = dict(leave_target="hold_penalty", hold_complete='reward', start_pause="pause"),
        hold_penalty = dict(hold_penalty_end="wait", start_pause="pause", end_state=True),
        reward = dict(reward_end="wait", start_pause="pause", stoppable=False, end_state=True),
        pause = dict(end_pause="wait", end_state=True),
    )

    def _test_hold_complete(self, time_in_state):
        time_since_hold_start = self.get_time() - self.most_recent_hold_start_time
        return self.hold_time<=(time_since_hold_start)

    def _start_hold_flash_off(self):
        #Turn off the target
        #Which actual animatino blob do I hide?
        target = self.targets[1]
        #Show the target
        target.hide()
        self.sync_event('TARGET_OFF', self.gen_indices[self.target_index])

    def _test_flash_interval_complete(self, time_in_state):
        #wait until the inter flash interval has completed before turning the flash back on
        interval_complete = time_in_state>=self.target_flash_time_s[1]
        max_flashes_reached = self.flash_tracker >= self.total_flashes
        return interval_complete and ~max_flashes_reached

    def _test_flash_cycle_complete(self, time_in_state):
        #make sure we have not encroached on the end hold buffer time
        time_since_hold_start = self.get_time() - self.most_recent_hold_start_time
        time_elapsed_check = self.hold_time<=(time_since_hold_start + self.flash_buffer_time_s)
        flash_elapsed_check = self.flash_tracker>self.total_flashes

        return time_elapsed_check or flash_elapsed_check

    def _test_start_flash_cycle(self, time_in_state):
        #Test to see if the buffer period at the beginningof the hold has been completed so that we can start showing peripheral targets
        return time_in_state > self.flash_buffer_time_s
    
    def _test_flash_complete(self, time_in_state):
        return time_in_state>self.target_flash_time_s[0]

    def _start_hold_start_buffer(self):
        #Save the start time of the hold for future reference and checking
        self.most_recent_hold_start_time = self.get_time()
        self.flash_tracker = 0

    def _start_hold_flash_on(self):
        #We want to grab the first peripheral target and turn it on
        #Which actual animatino blob do I move? The second one
        target = self.targets[1] 

        #Increment the flash tracker
        self.flash_tracker += 1

        #Incrementing to the next target in the generator
        self.target_index += 1
        #move the animation object to the correct location
        target_position = self.targs[self.target_index] - self.offset_cube
        target.move_to_position(target_position)

        #Show the target
        target.show()
        #Sync to ecube
        self.sync_event('TARGET_ON', self.gen_indices[self.target_index])
        #Save the target location
        self.target_location = self.targs[self.target_index]

    def _start_target(self):
        #self.target_index += 1 #I'm not sure if this makes sense to increment here; may

        # Show target if it is hidden (this is the first target, or previous state was a penalty)
        #UPdate!
        target = self.targets[self.center_target_index]
        #if self.target_index == 0:
        target.move_to_position(self.center_target_position - self.offset_cube) #Check the formating on these indicies; self.targs[self.target_index] - self.offset_cube)
        target.show()
        self.sync_event('TARGET_ON', self.center_target_index) #Double check that 0 corresponds correctly to the center target
        self.target_location = self.center_target_position # save for BMILoop

    def _test_enter_target(self, ts):
        '''
        return true if the distance between center of cursor and target is smaller than the cursor radius
        '''
        eye_pos = self.calibrated_eye_pos
        target_pos = self.center_target_position[0:2] #Only grab two values from the target position to match the shape of the eye position
        d_eye = np.linalg.norm(eye_pos - target_pos)
        return (d_eye <= self.target_radius + self.fixation_radius_buffer)

    def _test_leave_target(self, ts):
        eye_pos = self.calibrated_eye_pos
        target_pos = self.center_target_position[0:2] #Only grab two values from the target position to match the shape of the eye position
        d_eye = np.linalg.norm(eye_pos - target_pos)
        return not (d_eye <= self.target_radius + self.fixation_radius_buffer)
    
    def _parse_next_trial(self):
        '''Check that the generator has the required data'''
        
        super()._parse_next_trial()
        assert self.gen_indices.shape[0]>=np.floor(self.total_flashes), f'The specified generator chain_length is shorter than the total number of flashes needed per trial. Please increase chain length to {np.floor(self.total_flashes)} or greater'

    def _end_hold_end_buffer(self):
        target = self.targets[self.center_target_index]
        target.hide()
        #self.sync_event('TARGET_OFF', self.center_target_index)
        
    @staticmethod  
    def centerout_2D_chain(nblocks=100, ntargets=8, chain_length=5, distance=10, origin=(0,0,0)):
        '''
        Chains of targets, starting at the center and then random 
         peripheral targets radially arranged around the center.

        Returns
        -------
        [nblocks*ntargets x 1] array of tuples containing trial indices and [chain_length x 3] target coordinates
        '''
        #Could I add a check here that looks at the total number of flashes and updates the generator here?
        gen = ScreenTargetCapture.out_2D(nblocks*chain_length, ntargets, distance, origin)
        for _ in range(nblocks*ntargets):
            targs = np.zeros([chain_length, 3]) + origin
            indices = np.zeros([chain_length,1])

            for chain_idx in range(chain_length):
                idx, pos = next(gen)
                targs[chain_idx,:] = pos[0]
                indices[chain_idx] = idx

            yield indices, targs