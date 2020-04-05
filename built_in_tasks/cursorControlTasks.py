
from .manualcontrolmultitasks import ManualControlMulti
from riglib.stereo_opengl.window import WindowDispl2D
from .bmimultitasks import BMIControlMulti
import pygame
import numpy as np
import copy

from riglib.bmi.extractor import DummyExtractor
from riglib.bmi.state_space_models import StateSpaceEndptVel2D
from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter

class CursorControl(ManualControlMulti, WindowDispl2D):
    '''
    this class implements a python cursor control task for human
    '''

    def __init__(self, *args, **kwargs):
        # just run the parent ManualControlMulti's initialization
        self.move_step = 1
        super(CursorControl, self).__init__(*args, **kwargs)

    def init(self):
        pygame.init()
        self.assist_level = (0.5, 0.5)
        super(CursorControl, self).init()

    # override the _cycle function
    def _cycle(self):
        #print(self.state)

        self.move_effector_cursor()
        super(CursorControl, self)._cycle()

    # do nothing
    def move_effector(self):
        pass

    def move_plant(self, **kwargs):
        pass

    # use keyboard to control the task
    def move_effector_cursor(self):
        np.array([0., 0., 0.])
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
        super(CursorControl, self)._start_wait()

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause
        