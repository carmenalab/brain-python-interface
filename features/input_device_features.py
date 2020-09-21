
import pygame
import numpy as np
import copy
from riglib.source import DataSourceSystem

class KeyboardControl(object):
    '''
    this class implements a python cursor control task for human
    '''

    def __init__(self, *args, **kwargs):
        self.move_step = 1
        self.assist_level = (0.5, 0.5)
        super(KeyboardControl, self).__init__(*args, **kwargs)
    
    # override the _cycle function
    def _cycle(self):
        self.move_effector_cursor()
        super(KeyboardControl, self)._cycle()

    def move_effector(self):
        pass

    def move_plant(self, **kwargs):
        pass

    # use keyboard to control the task
    def move_effector_cursor(self):
        curr_pos = copy.deepcopy(self.plant.get_endpoint_pos())

        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.type == pygame.K_q:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_LEFT:
                    curr_pos[0] -= self.move_step
                if event.key == pygame.K_RIGHT:
                    curr_pos[0] += self.move_step
                if event.key == pygame.K_UP:
                    curr_pos[2] += self.move_step
                if event.key == pygame.K_DOWN:
                    curr_pos[2] -= self.move_step
            #print('Current position: ')
            #print(curr_pos)

        # set the current position
        self.plant.set_endpoint_pos(curr_pos)

    def _start_wait(self):
        self.wait_time = 0.
        super(KeyboardControl, self)._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause

class MouseControl(KeyboardControl):

    def init(self, *args, **kwargs):
        self.pos = self.plant.get_endpoint_pos()
        super(MouseControl, self).init(*args, **kwargs)
    
    def move_effector_cursor(self):

        # Update position but keep mouse in center
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        rel = pygame.mouse.get_rel()
        self.pos[0] += rel[0] / self.window_size[0] * self.fov
        self.pos[2] -= rel[1] / self.window_size[1] * self.fov
        pos, _ = self.plant._bound(self.pos, [])
        self.plant.set_endpoint_pos(pos)

    def cleanup(self, *args, **kwargs):
        pygame.mouse.set_visible(True)
        super(MouseControl, self).cleanup(*args, **kwargs)

from .neural_sys_features import CorticalBMI
class MouseBMI(CorticalBMI):
    @property 
    def sys_module(self):
        from riglib import debug
        return debug