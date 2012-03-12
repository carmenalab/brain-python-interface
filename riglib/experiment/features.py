import random
import pygame
import traceback

from . import traits
from riglib import calibrations

class RewardSystem(traits.HasTraits):
    '''Use the reward system during the reward phase'''
    def __init__(self, *args, **kwargs):
        from riglib import reward
        super(RewardSystem, self).__init__(*args, **kwargs)
        self.reward = reward.open()

    def _start_reward(self):
        if self.reward is not None:
            self.reward.reward(self.reward_time*1000.)

class Autostart(traits.HasTraits):
    '''Automatically begins the trial from the wait state, with a random interval drawn from `rand_start`'''
    rand_start = traits.Array(value=(0.5, 2.), shape=(2,), desc="Start interval")

    def __init__(self, *args, **kwargs):
        self.pause = False
        super(Autostart, self).__init__(*args, **kwargs)

    def _start_wait(self):
        s, e = self.rand_start
        self.wait_time = random.random()*(e-s) + s
        
    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause
    
    def _test_premature(self, ts):
        return self.event is not None

class Button(object):
    '''Adds the ability to respond to the button, as well as to keyboard responses'''
    def screen_init(self):
        super(Button, self).screen_init()
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def _get_event(self):
        btnmap = {1:1, 3:4}
        for btn in pygame.event.get(pygame.MOUSEBUTTONDOWN):
            if btn.button in btnmap:
                return btnmap[btn.button]

        return super(Button, self)._get_event()
    
    def _while_reward(self):
        super(Button, self)._while_reward()
        pygame.event.clear()
    
    def _while_penalty(self):
        #Clear out the button buffers
        super(Button, self)._while_penalty()
        pygame.event.clear()
    
    def _while_wait(self):
        super(Button, self)._while_wait()
        pygame.event.clear()

class IgnoreCorrectness(object):
    '''Allows any response to be correct, not just the one defined. Overrides for trialtypes'''
    def __init__(self, *args, **kwargs):
        super(IgnoreCorrectness, self).__init__(*args, **kwargs)
        if hasattr(self, "trial_types"):
            for ttype in self.trial_types:
                del self.status[ttype]["%s_correct"%ttype]
                del self.status[ttype]["%s_incorrect"%ttype]
                self.status[ttype]["correct"] = "reward"
                self.status[ttype]["incorrect"] = "penalty"

    def _test_correct(self, ts):
        return self.event is not None

    def _test_incorrect(self, ts):
        return False

class EyeData(object):
    def __init__(self, *args, **kwargs):
        from riglib import shm
        super(EyeData, self).__init__(*args, **kwargs)
        self.eyedata = shm.EyeData()

    def run(self):
        self.eyedata.start()
        super(EyeData, self).run()
    
    def _start_None(self):
        self.eyedata.stop()
        super(EyeData, self)._start_None()

class CalibratedEyeData(traits.HasTraits):
    cal_profile = traits.Instance(calibrations.Profile)

    def __init__(self, *args, **kwargs):
        from riglib import shm
        super(CalibratedEyeData, self).__init__(*args, **kwargs)
        self.eyedata = shm.EyeData()
        self.eyedata.filter = self.cal_profile
    
    def run(self):
        self.eyedata.start()
        super(CalibratedEyeData, self).run()
    
    def _start_None(self):
        self.eyedata.stop()
        super(CalibratedEyeData, self)._start_None()

class SimulatedEyeData(traits.HasTraits):
    fixations = traits.Array(value=[(0,0), (-0.6,0.3), (0.6,0.3)], desc="Location of fixation points")
    fixation_len = traits.Float(0.5, desc="Length of a fixation")

    def __init__(self, *args, **kwargs):
        from riglib import shm
        super(SimulatedEyeData, self).__init__(*args, **kwargs)
        self.eyedata = shm.EyeSimulate(fixations=self.fixations, isi=self.fixation_len*1e3)

    def run(self):
        self.eyedata.start()
        super(SimulatedEyeData, self).run()
    
    def _start_None(self):
        self.eyedata.stop()
        super(SimulatedEyeData, self)._start_None()

class MotionData(traits.HasTraits):
    marker_count = traits.Int(8, desc="Number of markers to return")

    def __init__(self, *args, **kwargs):
        from riglib import shm
        super(EyeData, self).__init__(*args, **kwargs)
        self.motiondata = shm.MotionData(marker_count=self.marker_count)
        
    def run(self):
        self.motiondata.start()
        super(MotionData, self).run()
    
    def _start_None(self):
        self.eyedata.stop()
        super(MotionData, self)._start_None()
