'''
Base tasks for generic point-to-point reaching
'''
import numpy as np
from collections import OrderedDict
import time
import os
import math
import traceback

from riglib.stereo_opengl.primitives import Sphere, Cube
from riglib.stereo_opengl.window import Window, FPScontrol, WindowDispl2D
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere, Cube
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex


####### CONSTANTS
sec_per_min = 60.0
RED = (1,0,0,.5)
GREEN = (0,1,0,0.5)
GOLD = (1., 0.843, 0., 0.5)
mm_per_cm = 1./10

class CircularTarget(object):
    def __init__(self, target_radius=2, target_color=(1, 0, 0, .5), starting_pos=np.zeros(3)):
        self.target_color = target_color
        self.default_target_color = tuple(self.target_color)
        self.target_radius = target_radius
        self.target_color = target_color
        self.position = starting_pos
        self.int_position = starting_pos
        self._pickle_init()

    def _pickle_init(self):
        self.sphere = Sphere(radius=self.target_radius, color=self.target_color)
        self.graphics_models = [self.sphere]
        self.sphere.translate(*self.position)

    def move_to_position(self, new_pos):
        self.int_position = new_pos
        self.drive_to_new_pos()

    def drive_to_new_pos(self):
        raise NotImplementedError

class VirtualCircularTarget(CircularTarget):
    def drive_to_new_pos(self):
        self.position = self.int_position
        self.sphere.translate(*self.position, reset=True)

    def hide(self):
        self.sphere.detach()

    def show(self):
        self.sphere.attach()

    def cue_trial_start(self):
        self.sphere.color = RED
        self.show()

    def cue_trial_end_success(self):
        self.sphere.color = GREEN

    def cue_trial_end_failure(self):
        self.sphere.color = RED
        self.hide()
        # self.sphere.color = GREEN
    def turn_yellow(self):
        self.sphere.color = GOLD

    def idle(self):
        self.sphere.color = RED
        self.hide()

    def pt_inside(self, pt):
        '''
        Test if a point is inside the target
        '''
        pos = self.sphere.xfm.move
        return (np.abs(pt[0] - pos[0]) < self.target_radius) and (np.abs(pt[2] - pos[2]) < self.target_radius)

    def reset(self):
        self.sphere.color = self.default_target_color

    def get_position(self):
        return self.sphere.xfm.move

class RectangularTarget(object):
    def __init__(self, target_width=4, target_height=4, target_color=(1, 0, 0, .5), starting_pos=np.zeros(3)):
        self.target_width = target_width
        self.target_height = target_height
        self.target_color = target_color
        self.default_target_color = tuple(self.target_color)
        self.position = starting_pos
        self.int_position = starting_pos
        self._pickle_init()

    def _pickle_init(self):
        self.cube = Cube(side_len=self.target_width, color=self.target_color)
        self.graphics_models = [self.cube]
        self.cube.translate(*self.position)
        #self.center_offset = np.array([self.target_width, 0, self.target_width], dtype=np.float64) / 2
        self.center_offset = np.array([0, 0, self.target_width], dtype=np.float64) / 2
    def move_to_position(self, new_pos):
        self.int_position = new_pos
        self.drive_to_new_pos()

    def drive_to_new_pos(self):
        raise NotImplementedError


class VirtualRectangularTarget(RectangularTarget):
    def drive_to_new_pos(self):
        self.position = self.int_position
        corner_pos = self.position - self.center_offset
        self.cube.translate(*corner_pos, reset=True)

    def hide(self):
        self.cube.detach()

    def show(self):
        self.cube.attach()

    def cue_trial_start(self):
        self.cube.color = RED
        self.show()

    def cue_trial_end_success(self):
        self.cube.color = GREEN

    def cue_trial_end_failure(self):
        self.cube.color = RED
        self.hide()

    def idle(self):
        self.cube.color = RED
        self.hide()

    def pt_inside(self, pt):
        '''
        Test if a point is inside the target
        '''
        pos = self.cube.xfm.move + self.center_offset
        # TODO this currently assumes that the cube doesn't rotate
        # print (pt[0] - pos[0]), (pt[2] - pos[2])
        return (np.abs(pt[0] - pos[0]) < self.target_width/2) and (np.abs(pt[2] - pos[2]) < self.target_height/2)

    def reset(self):
        self.cube.color = self.default_target_color

    def get_position(self):
        return self.cube.xfm.move
