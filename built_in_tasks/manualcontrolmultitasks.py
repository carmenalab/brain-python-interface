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
from .target_capture_task import ScreenTargetCapture

transformations = dict(
    none = np.identity(4),
)

class ManualControl(ScreenTargetCapture):
    '''Target capture task where the subject operates a joystick
    to control a cursor. Targets are captured by having the cursor
    dwell in the screen target for the allotted time'''

    # Settable Traits
    velocity_control = traits.Bool(False, desc="Position or velocity control")
    random_rewards = traits.Bool(False, desc="Add randomness to reward")
    transformation = traits.OptionsList(tuple(transformations.keys()), desc="Control transformation matrix")
    is_bmi_seed = True

    def __init__(self, *args, **kwargs):
        super(ManualControl, self).__init__(*args, **kwargs)
        self.current_pt=np.zeros([3]) #keep track of current pt
        self.last_pt=np.zeros([3]) #keep track of last pt to calc. velocity

    def update_report_stats(self):
        super(ManualControl, self).update_report_stats()
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

    def move_effector(self):
        ''' Returns the 3D coordinates of the cursor. For manual control, uses
        motiontracker / joystick / mouse data. If no data available, returns None'''

        if not hasattr(self, 'joystick'):
            return
        pt = self.joystick.get()
        if len(pt) == 0:
            return

        pt = pt[-1] # Use only the latest coordinate

        if len(pt) == 2:
            pt = np.array([pt[1], 0, pt[0], 1])
        else:
            pt = np.array([pt[0], pt[1], pt[2], 1])
        
        pt = np.matmul(transformations[self.transformation], pt)

        if not self.velocity_control:
            self.current_pt = pt[0:3]
        else:
            epsilon = 2*(10**-2) # Define epsilon to stabilize cursor movement
            if sum((pt[0:3])**2) > epsilon:

                # Add the velocity (units/s) to the position (units)
                self.current_pt = pt[0:3] / self.fps + self.last_pt
            else:
                self.current_pt = self.last_pt

        self.plant.set_endpoint_pos(self.current_pt)
        self.last_pt = self.plant.get_endpoint_pos().copy()

    @classmethod
    def get_desc(cls, params, report):
        duration = report[-1][-1] - report[0][-1]
        reward_count = 0
        for item in report:
            if item[0] == "reward":
                reward_count += 1
        return "{} rewarded trials in {} min".format(reward_count, duration)


class ManualControl2DWindow(ManualControl, WindowDispl2D):
    '''Seems redundant with 2D display feature. Not used any more'''

    fps = 20.
    def __init__(self,*args, **kwargs):
        super(ManualControl2DWindow, self).__init__(*args, **kwargs)

    def _start_wait(self):
        self.wait_time = 0.
        super(ManualControl2DWindow, self)._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

