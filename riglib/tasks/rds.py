import time
import pygame
import numpy as np

from .. import reward
from ..experiment import Pygame, TrialTypes, traits

class RDS(TrialTypes, Pygame.Pygame):
    ndots = traits.Int(250, desc="Number of dots on sphere")
    sphere_radius = traits.Float(250, desc="Radius of virtual sphere")
    dot_radius = traits.Int(5, desc="Radius of dots drawn on screen")
    sphere_speed = traits.Float(0.005*np.pi, desc="Speed at which the virtual sphere turns")
    disparity = traits.Float(.05, desc="Amount of disparity")

    trial_types = ["cw", "ccw"]
    trial_probs = traits.Array(value=[0.5, 0.5])

    def __init__(self, **kwargs):
        super(RDS, self).__init__(**kwargs)
        Pygame.Pygame.__init__(self, **kwargs)
        self.width, self.height = self.surf.get_size()
        self.loff = self.width / 4., self.height / 2.
        self.roff = self.width * 0.75, self.height / 2.
    
    def _init_sphere(self):
        u, v = np.random.rand(2, self.ndots)
        self._sphere = np.array([2*np.pi*u, np.arccos(2*v-1)])

    def _project_sphere(self, offset=True):
        theta, phi = self._sphere
        x = self.sphere_radius * np.cos(theta) * np.sin(phi)
        y = self.sphere_radius * np.sin(theta) * np.sin(phi)
        z = self.sphere_radius * np.cos(phi)
        d = y * self.disparity

        return np.array([x+d*(-1,1)[offset], z]).T
    
    def _draw_sphere(self):
        self.surf.fill(self.background)
        for pt in (self.loff + self._project_sphere(True)).astype(int):
            pygame.draw.circle(self.surf, (255, 255, 255), pt, self.dot_radius)
        
        for pt in (self.roff + self._project_sphere(False)).astype(int):
            pygame.draw.circle(self.surf, (255, 255, 255), pt, self.dot_radius)
        self.flip_wait()
    
    def _while_penalty(self):
        self.surf.fill((181,0,45))
        pygame.display.flip()
    
    def _start_reward(self):
        if reward is not None:
            reward.reward(self.reward_time*1000.)
    
    def _start_cw(self):
        self._init_sphere()

    def _start_ccw(self):
        self._init_sphere()
    
    def _while_cw(self):
        self._sphere[0] += self.sphere_speed
        self._draw_sphere()
    
    def _while_ccw(self):
        self._sphere[0] -= self.sphere_speed
        self._draw_sphere()
    
    def _test_cw_correct(self, ts):
        return ts > self.ignore_time
    
    def _test_cw_incorrect(self, ts):
        return self.event is not None
    
    def _test_ccw_correct(self, ts):
        return self.event is not None