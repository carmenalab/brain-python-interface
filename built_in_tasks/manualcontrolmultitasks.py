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

from .plantlist import plantlist
from .target_graphics import *
from .target_capture_task import ScreenTargetCapture


class JoystickMulti(ScreenTargetCapture):
    '''Target capture task where the subject operates a joystick
    to control a cursor. Targets are captured by having the cursor
    dwell in the screen target for the allotted time'''

    # Settable Traits
    joystick_method = traits.Float(1,desc="1: Normal velocity, 0: Position control")
    random_rewards = traits.Float(0,desc="Add randomness to reward, 1: yes, 0: no")
    joystick_speed = traits.Float(20, desc="Radius of cursor")

    is_bmi_seed = True

    def __init__(self, *args, **kwargs):
        super(JoystickMulti, self).__init__(*args, **kwargs)
        self.current_pt=np.zeros([3]) #keep track of current pt
        self.last_pt=np.zeros([3]) #keep track of last pt to calc. velocity

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

