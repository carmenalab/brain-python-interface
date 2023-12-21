'''
Force-based size task
'''
import numpy as np

from riglib.experiment import traits, LogExperiment
from riglib.stereo_opengl.window import Window
from .target_graphics import *

disk_colors = {
    'target': (1, 1, 1, 0.75),
    'cursor': (1, 0.5, 0, 0.5),
}

class DiskMatching(Window, LogExperiment):
    """Concrete implementation of TargetCapture task where targets
    are acquired by "holding" a cursor in an on-screen target"""

    status = dict(
        wait = dict(start_trial="target"),
        target = dict(enter_target="hold"),
        hold = dict(leave_target="hold_penalty", hold_complete="reward"),
        hold_penalty = dict(hold_penalty_end="target", end_state=True),
        reward = dict(reward_end="wait", stoppable=False, end_state=True)
    )

    # initial state
    state = "wait"

    # Runtime settable traits
    disk_cursor_offset = traits.Float(-0.09, desc="offset from raw joystick value to disk cursor radius")
    disk_cursor_gain = traits.Float(100., desc="conversion from raw joystick value to disk cursor radius")
    disk_radius = traits.Tuple((2.5, 5), desc="(min, max) radius (in cm) of the disk target")
    disk_tolerance = traits.Float(0.5, desc="allowed cm difference in radius between disk cursor and disk target")
    disk_target_color = traits.OptionsList("target", *disk_colors, desc="Color of the disk target", bmi3d_input_options=list(disk_colors.keys()))
    disk_cursor_color = traits.OptionsList("cursor", *disk_colors, desc='Color of disk cursor ', bmi3d_input_options=list(disk_colors.keys()))
    disk_pos = traits.Tuple((0., 0., 0.), desc='Where to locate the disks') 

    hold_time = traits.Tuple((0.5, 1.5), desc="Amount of time in (min, max) seconds the target must be held before reward")
    hold_penalty_time = traits.Float(0.5, desc="Amount of time in seconds spent in hold penalty")
    reward_time = traits.Float(.5, desc="Length of reward dispensation")

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.disk_cursor_pos = self.disk_pos
        self.disk_target_pos = np.array(self.disk_pos).copy()
        self.disk_target_pos[1] = 10 # since targets are actually spheres, move them far apart for 2D screen
        self.disk_cursor = VirtualCircularTarget(target_radius=0, target_color=disk_colors[self.disk_cursor_color], starting_pos=self.disk_cursor_pos)
        for model in self.disk_cursor.graphics_models:
            self.add_model(model)


    def init(self):
        self.add_dtype('disk_cursor_size', 'f8', (1,))
        self.add_dtype('disk_target_size', 'f8', (1,))
        super().init()

    def update_disk_cursor(self):
        
        # Remove the previous disk cursor
        for model in self.disk_cursor.graphics_models:
            self.remove_model(model)

        # Add a new cursor with updated radius
        raw_value = self.joystick.get()
        if raw_value is None:
            raw_value = 0.09
        cm_value = (raw_value + self.disk_cursor_offset) * self.disk_cursor_gain
        max_radius = 8
        min_radius = 0.5
        self.disk_cursor_radius = min(max(cm_value, min_radius), max_radius)
        self.task_data['disk_cursor_size'] = self.disk_cursor_radius
        self.disk_cursor = VirtualCircularTarget(target_radius=self.disk_cursor_radius, target_color=disk_colors[self.disk_cursor_color], starting_pos=self.disk_cursor_pos)
        for model in self.disk_cursor.graphics_models:
            self.add_model(model)

    def _cycle(self):
        '''
        Calls any update functions necessary and redraws screen
        '''
        self.update_disk_cursor()
        super()._cycle()

   
    #### TEST FUNCTIONS ####
    def _test_start_trial(self, ts):
        return True

    def _test_enter_target(self, ts):
        d = abs(self.disk_cursor_radius - self.disk_target_radius)
        return d <= self.disk_tolerance

    def _test_leave_target(self, ts):
        d = abs(self.disk_cursor_radius - self.disk_target_radius)
        return d > self.disk_tolerance

    def _test_hold_complete(self, ts):
        return ts > self.hold_time_trial
    
    def _test_hold_penalty_end(self, ts):
        return ts > self.hold_penalty_time

    def _test_reward_end(self, time_in_state):
        return time_in_state > self.reward_time

    ### STATE FUNCTIONS ###
    def _start_wait(self):

        # Select a random parameters
        self.disk_target_radius = np.random.uniform(*self.disk_radius)
        self.hold_time_trial = np.random.uniform(*self.hold_time)
        print("New target radius:", self.disk_target_radius)

    def _start_target(self):
        self.sync_event('TARGET_ON', 0)
        
        self.task_data['disk_target_size'] = self.disk_target_radius
        self.disk_target = VirtualCircularTarget(target_radius=self.disk_target_radius, target_color=disk_colors[self.disk_target_color], starting_pos=self.disk_target_pos)
        for model in self.disk_target.graphics_models:
            self.add_model(model)

    def _start_hold(self):
        self.sync_event('CURSOR_ENTER_TARGET', 0)

    def _start_hold_penalty(self):
        self.sync_event('HOLD_PENALTY')

        # Remove the disk target
        for model in self.disk_target.graphics_models:
            self.remove_model(model)


    def _start_reward(self):
        self.disk_target.cue_trial_end_success()
        self.sync_event('REWARD')
        
    def _end_reward(self):
        self.sync_event('TRIAL_END')

        # Remove the disk target
        for model in self.disk_target.graphics_models:
            self.remove_model(model)
