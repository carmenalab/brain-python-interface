#!/usr/bin/python
from riglib.stereo_opengl import window
from riglib.stereo_opengl.primitives import Sphere
from riglib.experiment import traits, Sequence
from riglib.stereo_opengl.render import stereo, Renderer

from riglib.stereo_opengl.window import Window

import pygame
import numpy as np
import importlib

#importlib.reload(window)
m_to_cm = 100
target_pos_radius = 10

class TestGraphics(Sequence, Window):
    status = dict(
        wait = dict(stop=None),
    )   

    #initial state
    state = "wait"
    target_radius = 2.

    
        
    #create targets, cursor objects, initialize
    def __init__(self, *args, **kwargs):
        # Add the target and cursor locations to the task data to be saved to
        # file
        #super(TestGraphics, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.dtype = [('target', 'f', (3,)), ('cursor', 'f', (3,)), (('target_index', 'i', (1,)))]
        self.target1 = Sphere(radius=self.target_radius, color=(1,0,0,.5))
        self.add_model(self.target1)
        self.target2 = Sphere(radius=self.target_radius, color=(1,0,0,0.5))
        self.add_model(self.target2)
            
        # Initialize target location variable
        self.target_location = np.array([0.0,0.0,0.0])

    ##### HELPER AND UPDATE FUNCTIONS ####

#<<<<<<< HEAD
    def _get_renderer(self):
        return stereo.MirrorDisplay(self.window_size, self.fov, 1, 1024, self.screen_dist, self.iod)
    def _cycle(self):
        
        super()._cycle()

    #### STATE FUNCTIONS ####
    def _while_wait(self):
        #print("_while_wait")
        
        delta_movement = np.array([0,0,0.01])
        self.target_location += delta_movement

        self.target1.translate(self.target_location[0], 
                               self.target_location[1],
                               self.target_location[2], reset=True)
        self.requeue()
        self.draw_world()
        print('current target 1 position ' + np.array2string(self.target_location))


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
        targ = m_to_cm*targets[target_inds[k], :]
        yield np.array([[center[0], 0, center[1]],
                        [targ[0], 0, targ[1]]])
        k += 1


if __name__ == "__main__":
    print('Remember to set window size in stereoOpenGL class')
    gen = target_seq_generator(8, 1000)
    w = TestGraphics(gen)
    w.init()
    w.run()
    
 

