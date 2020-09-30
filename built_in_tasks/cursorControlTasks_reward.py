"""
This script is a modified version of cursorControlTasks_optitrack to include reward feature in cursor control task

modified by Pavi Aug 2020
"""

from manualcontrolmultitasks import ManualControlMulti
from riglib.stereo_opengl.window import WindowDispl2D
# from bmimultitasks import BMIControlMulti
import pygame
import numpy as np
import copy

# from riglib.bmi.extractor import DummyExtractor
# from riglib.bmi.state_space_models import StateSpaceEndptVel2D
# from riglib.bmi.bmi import Decoder, BMISystem, GaussianStateHMM, BMILoop, GaussianState, MachineOnlyFilter
from riglib import experiment
from features.hdf_features import SaveHDF
from features.reward_features import RewardSystem

class CursorControl(ManualControlMulti, WindowDispl2D):
    '''
    this class implements a python cursor control task for human
    '''

    def __init__(self, *args, **kwargs):
        # just run the parent ManualControlMulti's initialization
        self.move_step = 1

        # Initialize target location variable
        #target location and index have been initializd

        super(CursorControl, self).__init__(*args, **kwargs)

    def init(self):
        pygame.init()
        self.assist_level = (0, 0)
        super(CursorControl, self).init()

    # override the _cycle function
    def _cycle(self):
        #print(self.state)

        #target and plant data have been saved in
        #the parent manualcontrolmultitasks

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

#this task can be run on its
#we will not involve  database at this time
target_pos_radius = 10

def target_seq_generator(n_targs, n_trials):
    #generate targets
    angles = np.transpose(np.arange(0,2*np.pi,2*np.pi / n_targs))
    unit_targets = targets = np.stack((np.cos(angles), np.sin(angles)),1)
    targets = unit_targets * target_pos_radius

    center = np.array((0,0))

    target_inds = np.random.randint(0, n_targs, n_trials)
    target_inds[0:n_targs] = np.arange(min(n_targs, n_trials))

    k = 0
    while k < n_trials:
        targ = targets[target_inds[k], :]
        yield np.array([[center[0], 0, center[1]],
                        [targ[0], 0, targ[1]]])
        k += 1



if __name__ == "__main__":
    print('Remember to set window size in stereoOpenGL class')
    gen = target_seq_generator(8, 2)

    # incorporate the saveHDF feature by blending code
    # see tests\start_From_cmd_line_sim
    from features.reward_features import RewardSystem
    #from features.optitrack_feature import MotionData
    from features.hdf_features import SaveHDF

    base_class = CursorControl
    feats = [RewardSystem, SaveHDF]
    Exp = experiment.make(base_class, feats=feats)
    print(Exp)

    exp = Exp(gen)
    exp.init()
    exp.run()  # start the task