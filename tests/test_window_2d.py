#!/usr/bin/python
from riglib.stereo_opengl import window
from riglib.stereo_opengl.primitives import Sphere
from riglib.experiment import traits, Sequence
from riglib.stereo_opengl.render import stereo, Renderer

from riglib.stereo_opengl.window import Window

import pygame
import numpy as np

reload(window)

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
        super(TestGraphics, self).__init__(*args, **kwargs)
        self.dtype = [('target', 'f', (3,)), ('cursor', 'f', (3,)), (('target_index', 'i', (1,)))]
        self.target1 = Sphere(radius=self.target_radius, color=(1,0,0,.5))
        self.add_model(self.target1)
        self.target2 = Sphere(radius=self.target_radius, color=(1,0,0,.5))
        self.add_model(self.target2)
            
        # Initialize target location variable
        self.target_location = np.array([0,0,0])

    ##### HELPER AND UPDATE FUNCTIONS ####

<<<<<<< HEAD
    def _get_renderer(self):
        return stereo.MirrorDisplay(self.window_size, self.fov, 1, 1024, self.screen_dist, self.iod)

    #### STATE FUNCTIONS ####
    def _while_wait(self):
        print "_while_wait"
        self.target1.translate(0, 0, 0, reset=True)
        self.target1.attach()
        self.requeue()
        self.draw_world()


def target_seq_generator(n_targs, n_trials):
    target_inds = np.random.randint(0, n_targs, n_trials)
    target_inds[0:n_targs] = np.arange(min(n_targs, n_trials))
    k = 0
    while k < n_trials:
        targ = m_to_cm*targets[target_inds[k], :]
        yield np.array([[center[0], 0, center[1]],
                        [targ[0], 0, targ[1]]])
        k += 1

gen = target_seq_generator(8, 1000)
w = TestGraphics(gen)
w.init()
w.run()
