'''
Virtual target capture tasks where cursors are controlled by physical
motion interfaces such as joysticks
'''
import numpy as np
from collections import OrderedDict
import time
import os
import math
import traceback

from riglib.experiment import traits

from .target_graphics import *
from .target_capture_task import ScreenTargetCapture, ScreenReachAngle
from riglib.stereo_opengl.window import WindowDispl2D


rotations = dict(
    yzx = np.array(
        [[0, 1, 0, 0], 
        [0, 0, 1, 0], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1]]
    ),
    zyx = np.array(
        [[0, 0, 1, 0], 
        [0, 1, 0, 0], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1]]
    ),
    xzy = np.array(
        [[1, 0, 0, 0], 
        [0, 0, 1, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1]]
    ),
    xyz = np.identity(4),
)

class ManualControlMixin(traits.HasTraits):
    '''Target capture task where the subject operates a joystick
    to control a cursor. Targets are captured by having the cursor
    dwell in the screen target for the allotted time'''

    # Settable Traits
    wait_time = traits.Float(2., desc="Time between successful trials")
    velocity_control = traits.Bool(False, desc="Position or velocity control")
    random_rewards = traits.Bool(False, desc="Add randomness to reward")
    rotation = traits.OptionsList(*rotations, desc="Control rotation matrix", bmi3d_input_options=list(rotations.keys()))
    scale = traits.Float(1.0, desc="Control scale factor")
    offset = traits.Array(value=[0,0,0], desc="Control offset")
    is_bmi_seed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_pt=np.zeros([3]) #keep track of current pt
        self.last_pt=self.starting_pos #keep track of last pt to calc. velocity
        self._quality_window_size = 500 # how many cycles to accumulate quality statistics
        self.reportstats['Input quality'] = "100 %"
        if self.random_rewards:
            self.reward_time_base = self.reward_time

    def init(self):
        self.add_dtype('manual_input', 'f8', (3,))
        super().init()
        self.no_data_counter = np.zeros((self._quality_window_size,), dtype='?')

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

    def _test_trial_complete(self, ts):
        if self.target_index==self.chain_length-1 :
            if self.random_rewards:
                if not self.rand_reward_set_flag: #reward time has not been set for this iteration
                    self.reward_time = np.max([2*(np.random.rand()-0.5) + self.reward_time_base, self.reward_time_base/2]) #set randomly with min of base / 2
                    self.rand_reward_set_flag =1
                    #print self.reward_time, self.rand_reward_set_flag
            return self.target_index==self.chain_length-1

    def _test_reward_end(self, ts):
        #When finished reward, reset flag.
        if self.random_rewards:
            if ts > self.reward_time:
                self.rand_reward_set_flag = 0
                #print self.reward_time, self.rand_reward_set_flag, ts
        return ts > self.reward_time

    def _transform_coords(self, coords):
        ''' 
        Returns transformed coordinates based on rotation, offset, and scale traits
        '''
        offset = np.array(
            [[1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0], 
            [self.offset[0], self.offset[1], self.offset[2], 1]]
        )
        scale = np.array(
            [[self.scale, 0, 0, 0], 
            [0, self.scale, 0, 0], 
            [0, 0, self.scale, 0], 
            [0, 0, 0, 1]]
        )
        old = np.concatenate((np.reshape(coords, -1), [1]))
        new = np.linalg.multi_dot((old, offset, scale, rotations[self.rotation]))
        return new[0:3]

    def _get_manual_position(self):
        '''
        Fetches joystick position
        '''
        if not hasattr(self, 'joystick'):
            return
        pt = self.joystick.get()
        if len(pt) == 0:
            return

        pt = pt[-1] # Use only the latest coordinate

        if len(pt) == 2:
            pt = np.concatenate((np.reshape(pt, -1), [0]))

        return [pt]

    def move_effector(self):
        ''' 
        Sets the 3D coordinates of the cursor. For manual control, uses
        motiontracker / joystick / mouse data. If no data available, returns None
        '''

        # Get raw input and save it as task data
        raw_coords = self._get_manual_position() # array of [3x1] arrays
        if raw_coords is None or len(raw_coords) < 1:
            self.no_data_counter[self.cycle_count % self._quality_window_size] = 1
            self.update_report_stats()
            self.task_data['manual_input'] = np.empty((3,))
            return

        self.task_data['manual_input'] = raw_coords.copy()
        self.no_data_counter[self.cycle_count % self._quality_window_size] = 0

        # Transform coordinates
        coords = self._transform_coords(raw_coords)
        if self.limit2d:
            coords[1] = 0

        # Set cursor position
        if not self.velocity_control:
            self.current_pt = coords
        else:
            epsilon = 2*(10**-2) # Define epsilon to stabilize cursor movement
            if sum((coords)**2) > epsilon:

                # Add the velocity (units/s) to the position (units)
                self.current_pt = coords / self.fps + self.last_pt
            else:
                self.current_pt = self.last_pt

        self.plant.set_endpoint_pos(self.current_pt)
        self.last_pt = self.plant.get_endpoint_pos().copy() # bound position

    def update_report_stats(self):
        super().update_report_stats()
        window_size = min(max(1, self.cycle_count), self._quality_window_size)
        num_missing = np.sum(self.no_data_counter[:window_size])
        quality = 1 - num_missing / window_size
        self.reportstats['Input quality'] = "{} %".format(int(100*quality))

    @classmethod
    def get_desc(cls, params, log_summary):
        duration = round(log_summary['runtime'] / 60, 1)
        return "{}/{} succesful trials in {} min".format(
            log_summary['n_success_trials'], log_summary['n_trials'], duration)


class ManualControl(ManualControlMixin, ScreenTargetCapture):
    '''
    Slightly refactored original manual control task
    '''
    pass

class ManualControlDirectionConstraint(ManualControlMixin, ScreenReachAngle):
    '''
    Adds an additional constraint that the direction of travel must be within a certain angle
    '''
    pass