'''
Virtual target capture tasks where cursors are controlled by physical
motion interfaces such as joysticks
'''
import numpy as np
from scipy.spatial.transform import Rotation as R

from riglib.experiment import traits

from .target_graphics import *
from .target_capture_task import ScreenTargetCapture
from .target_capture_task_xt import ScreenReachAngle, ScreenReachLine, SequenceCapture, ScreenTargetCapture_ReadySet
from .target_capture_task_eye import EyeConstrainedTargetCapture, HandConstrainedEyeCapture, EyeConstrainedHandCapture, \
    EyeHandSequenceCapture, ScreenTargetCapture_Saccade, EyeHandCaptureBlock, EyeHandCaptureBlock_sequence, ScreenTargetCapture_Saccade_withBackgroundTargets
from .target_tracking_task import ScreenTargetTracking
from .rotation_matrices import *

class ManualControlMixin(traits.HasTraits):
    '''Target capture task where the subject operates a joystick
    to control a cursor. Targets are captured by having the cursor
    dwell in the screen target for the allotted time'''

    # Settable Traits
    wait_time = traits.Float(2., desc="Time between successful trials")
    velocity_control = traits.Bool(False, desc="Position or velocity control")
    random_rewards = traits.Bool(False, desc="Add randomness to reward")
    rotation = traits.OptionsList(*rotations, desc="Rotation to transform raw to bmi3d coordinates", bmi3d_input_options=list(rotations.keys()))
    scale = traits.Float(1.0, desc="Scale factor to transform raw to bmi3d coordinates")
    baseline_rotation = traits.OptionsList(*baseline_rotations, desc="Rotation to define the mapping between bmi3d coordinates and baseline workspace", bmi3d_input_options=list(baseline_rotations.keys()))
    exp_rotation = traits.OptionsList(*exp_rotations, desc="Experimental rotation to manipulate the mapping between baseline workspace and screen coordinates", bmi3d_input_options=list(exp_rotations.keys()))
    pertubation_rotation = traits.Float(0.0, desc="Experimental rotation about bmi3d y-axis in degrees") # this is perturbation_rotation_y
    perturbation_rotation_z = traits.Float(0.0, desc="Experimental rotation about bmi3d z-axis in degrees")
    perturbation_rotation_x = traits.Float(0.0, desc="Experimental rotation about bmi3d x-axis in degrees")
    offset = traits.Array(value=[0,0,0], desc="Offset to transform raw to bmi3d coordinates")
    exp_gain = traits.Float(1.0, desc="Experimental gain scale factor to manipulate the mapping to screen coordinates")
    is_bmi_seed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_pt = np.zeros([3]) # keep track of current pt
        self.last_pt = np.array(self.starting_pos) # keep track of last pt to calc. velocity
        self.prev_coords = None # keep track of last valid coords
        self._quality_window_size = 500 # how many cycles to accumulate quality statistics
        self.reportstats['Input quality'] = "100 %"
        if self.random_rewards:
            self.reward_time_base = self.reward_time

    def init(self):
        self.add_dtype('manual_input', 'f8', (3,))
        self.add_dtype('user_screen', 'f8', (3,))
        super().init()
        self.no_data_counter = np.zeros((self._quality_window_size,), dtype='?')

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

    # def _test_trial_complete(self, ts):
    #     if self.target_index==self.chain_length-1 :
    #         if self.random_rewards:
    #             if not self.rand_reward_set_flag: #reward time has not been set for this iteration
    #                 self.reward_time = np.max([2*(np.random.rand()-0.5) + self.reward_time_base, self.reward_time_base/2]) #set randomly with min of base / 2
    #                 self.rand_reward_set_flag =1
    #                 #print self.reward_time, self.rand_reward_set_flag
    #         return self.target_index==self.chain_length-1

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
        old = np.concatenate((np.reshape(coords, -1), [1])) # manual input (3,) plus offset term
        new = np.linalg.multi_dot((old, offset, scale, rotations[self.rotation], baseline_rotations[self.baseline_rotation], exp_rotations[self.exp_rotation])) # screen coords (3,) plus offset term
        pertubation_rot = R.from_euler('y', self.pertubation_rotation, degrees=True) # this is perturb_rot_y
        perturb_rot_z = R.from_euler('z', self.perturbation_rotation_z, degrees=True)
        perturb_rot_x = R.from_euler('x', self.perturbation_rotation_x, degrees=True)
        return np.linalg.multi_dot((new[0:3] * self.exp_gain, pertubation_rot.as_matrix(), perturb_rot_z.as_matrix(), perturb_rot_x.as_matrix()))

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

    def move_effector(self, pos_offset=[0,0,0], vel_offset=[0,0,0]):
        ''' 
        Sets the 3D coordinates of the cursor. For manual control, uses
        motiontracker / joystick / mouse data. If no data available, returns None
        '''

        # Get raw input from motor effector (e.g. optitrack, mouse, tablet touch app)
        raw_coords = self._get_manual_position() # array of [3x1] arrays

        if raw_coords is None or len(raw_coords) < 1: 
            # E.g. when mocap dots are outside optitrack's detectable region or when there is no touch input to the tablet  
            self.no_data_counter[self.cycle_count % self._quality_window_size] = 1
            self.update_report_stats()

            # Save manual input as NaN in task data
            self.task_data['manual_input'] = np.ones((3,))*np.nan
            self.task_data['user_screen'] = np.ones((3,))*np.nan

            # Use last valid coords to continue showing cursor at last valid position
            coords = self.prev_coords
            # print(coords)
            if coords is None:
                return

        else:
            # Save manual input in task data
            self.task_data['manual_input'] = raw_coords.copy()
            self.no_data_counter[self.cycle_count % self._quality_window_size] = 0

            # Transform coordinates to screen space and save in task data
            coords = self._transform_coords(raw_coords)
            self.task_data['user_screen'] = coords.copy()
        
            # Restrict cursor to only move along desired dimensions
            try:
                if self.limit2d:
                    coords[1] = 0
                if self.limit1d:
                    coords[1] = 0
                    coords[0] = 0
            except:
                if self.limit2d:
                    coords[1] = 0

            # Keep track of last valid coords
            self.prev_coords = coords
            # print(coords)

        # Add cursor disturbance
        final_coords = coords + pos_offset + vel_offset
        # print('final', coords, pos_offset, coords + pos_offset)
   
        # Finalize cursor position
        if not self.velocity_control:
            self.current_pt = final_coords
        
        else: # TODO how to implement velocity control?
            epsilon = 2*(10**-2) # Define epsilon to stabilize cursor movement
            if sum((final_coords)**2) > epsilon:
                # Add the velocity (units/s) to the position (units)
                self.current_pt = final_coords / self.fps + self.last_pt
            else:
                self.current_pt = self.last_pt

        # Move cursor (plant) to finalized position
        self.plant.set_endpoint_pos(self.current_pt)

        # Keep track of last valid cursor (plant) position
        self.last_pt = self.plant.get_endpoint_pos()

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

class ManualControlAreaConstraint(ManualControlMixin, ScreenReachLine):
    '''
    Adds an additional constraint that the cursor must be within a certain straight line
    '''
    pass

class TrackingTask(ManualControlMixin, ScreenTargetTracking):
    '''
    Track moving target task
    '''
    pass

class SequenceTask(ManualControlMixin, SequenceCapture):
    '''
    Sequence task. Subjects need to capture 3 targets. The last target appears while reaching to the 2st target.
    '''
    pass

class EyeConstrainedManualControl(ManualControlMixin, EyeConstrainedTargetCapture):
    '''
    Slightly refactored original manual control task
    '''
    pass

class HandConstrainedSaccadeTask(ManualControlMixin, HandConstrainedEyeCapture):
    '''
    Saccade task while holding different targets by hand
    '''
    pass

class EyeHandConstrainedReachingTask(ManualControlMixin, EyeHandCaptureBlock):
    '''
    Saccade and reaching task. The initial eye and hand positions are different. Subjects move their eyes or both eyes and hand to the goal target
    depending on trial blocks
    '''
    pass

class EyeHandConstrainedSequentialReachingTask(ManualControlMixin, EyeHandCaptureBlock_sequence):
    '''
    Saccade and reaching task. The task structure is similar to EyeHandCaputureBlock, but Subjects need to move their eyes and hand sequentially.
    '''
    pass

class EyeHandSequenceTask(ManualControlMixin, EyeHandSequenceCapture):
    '''
    Eye-hand sequence task where the hand go cue and the eye go cue come separately or simultaneously
    '''
    pass

class EyeConstrainedReachingTask(ManualControlMixin, EyeConstrainedHandCapture):
    '''
    Saccade and reaching task. The initial eye and hand positions are different, but the goal target is the same.
    '''
    pass

class SaccadeTask(ManualControlMixin, ScreenTargetCapture_Saccade):
    '''
    Center out saccade task. The controller for the cursor is eye positions. The target color changes when subjects fixate the target.
    '''
    pass




class SaccadeTaskWithBackgroundTargets(ManualControlMixin, ScreenTargetCapture_Saccade_withBackgroundTargets):

    '''
    Center out saccade task. The controller for the cursor is eye positions. The target color changes when subjects fixate the target.
    '''
    pass

class ReadySetGoTask(ManualControlMixin, ScreenTargetCapture_ReadySet):
    '''
    Center out task with ready set go auditory cues. Cues separated by 500 ms and participant is expected to move on final go cue. Additionally, participant must move out
    of center circle (mustmv_time) parameter or there will be an error. 
    '''
    pass