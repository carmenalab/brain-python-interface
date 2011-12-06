import time
import random
import pygame

from .. import gendots, reward
from ..experiment import Pygame, LogExperiment, TrialTypes, traits

class Dots(TrialTypes, Pygame.Pygame):
    trial_types = ["flat", "depth"]

    ignore_time = traits.Float(4.)

    def __init__(self, **kwargs):
        super(Dots, self).__init__(**kwargs)
        Pygame.Pygame.__init__(self, **kwargs)

        self.width, self.height = self.surf.get_size()
        mask = gendots.squaremask()
        mid = self.height / 2 - mask.shape[0] / 2
        lc = self.width / 4 - mask.shape[1] / 2
        rc = 3*self.width / 4 - mask.shape[1] / 2

        self.mask = mask
        self.coords = (lc, mid), (rc, mid)
    
    def _while_penalty(self):
        self.surf.fill((181,0,45))
        pygame.display.flip()
    
    def draw_frame(self):
        self.surf.blit(self.sleft, self.coords[0])
        self.surf.blit(self.sright, self.coords[1])
    
    def _start_reward(self):
        if reward is not None:
            reward.reward(self.reward_time*1000.)
    
    def _start_depth(self):
        left, right, flat = gendots.generate(self.mask)
        sleft = pygame.surfarray.make_surface((left>0).T)
        sright = pygame.surfarray.make_surface((right>0).T)
        sleft.set_palette([(0,0,0,255), (255,255,255,255)])
        sright.set_palette([(0,0,0,255), (255,255,255,255)])
        self.sleft, self.sright = sleft, sright

    def _start_flat(self):
        left, right, flat = gendots.generate(self.mask)
        sflat = pygame.surfarray.make_surface(flat>0)
        sflat.set_palette([(0,0,0,255), (255,255,255,255)])
        self.sflat = sflat
    
    def _while_depth(self):
        self.surf.blit(self.sleft, self.coords[0])
        self.surf.blit(self.sright, self.coords[1])
        self.flip_wait()
    
    def _while_flat(self):
        self.surf.blit(self.sflat, self.coords[0])
        self.surf.blit(self.sflat, self.coords[1])
        self.flip_wait()
    
    def _test_premature(self, ts):
        return self.event is not None
    
    def _test_flat_correct(self, ts):
        return ts > self.ignore_time
    
    def _test_flat_incorrect(self, ts):
        return self.event is not None
    
    def _test_depth_correct(self, ts):
        return self.event is not None
