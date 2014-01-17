from __future__ import division
import sys

import time
import numpy as np

import os
os.environ['DISPLAY'] = ':0'

from riglib.stereo_opengl.window import Window, FPScontrol
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere, Cone
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import ssao, stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex

from riglib.stereo_opengl.ik import RobotArm, RobotArm2D, RobotArm2J2D, RobotArm4J2D
from stereo_opengl.xfm import Quaternion
import time

from riglib.stereo_opengl.ik import RobotArm

import pygame

arm4j = RobotArm4J2D(link_radii = [.2,.2,.2,.2], joint_radii=[.2,.2,.2,.2])
cone = Sphere(radius=1)

class Test2(Window):

    def _start_draw(self):
        arm4j.set_joint_pos([0,0,np.pi/2,np.pi/2])
        arm4j.get_endpoint_pos()
        pass

    def _while_draw(self):
        ts = time.time() - self.start_time
        t = (ts / 2.) * np.pi   
        self.draw_world()

if __name__ == "__main__":
    win = Test2(window_size=(1920, 1080))
    #win.add_model(cone)
    win.add_model(arm4j)
    #win.screen_init()
    #win.draw_world()
    win.run()