import random
import pygame

from . import traits

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
    rand_start = traits.Tuple((1, 10))

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
    def _get_event(self):
        for btn in pygame.event.get(pygame.MOUSEBUTTONDOWN):
            return {1:1, 3:4}[btn.button]
                
        return super(Button, self)._get_event()
    
    def _while_penalty(self):
        #Clear out the button buffers
        pygame.event.pump()
        super(Button, self)._while_penalty()
    
    def _while_wait(self):
        pygame.event.pump()
        super(Button, self)._while_wait()

class ButtonOnly(Button):
    '''Forces the experiment to respond exclusively to the FTDI button, not to any keybooard events'''
    def _get_event(self):
        pygame.event.get(pygame.MOUSEBUTTONDOWN)

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
