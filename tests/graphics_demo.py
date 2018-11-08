
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

from riglib.stereo_opengl.ik import RobotArmGen2D
from riglib.stereo_opengl.xfm import Quaternion
import time

from riglib.stereo_opengl.ik import RobotArm

import pygame

arm4j = RobotArmGen2D(link_radii=.2, joint_radii=.2, link_lengths=[4,4,2,2])
cone = Sphere(radius=1)

pos_list = np.array([[0,0,0],[0,0,5]])

class Test2(Window):

    def __init__(self, *args, **kwargs):
        self.count=0
        super(Test2,self).__init__(*args, **kwargs)

    def _start_draw(self):
        #arm4j.set_joint_pos([0,0,np.pi/2,np.pi/2])
        #arm4j.get_endpoint_pos()
        pass

    def _while_draw(self):
        ts = time.time() - self.start_time
        t = (ts / 2.) * np.pi
        if self.count<len(pos_list):
            print("initial position = ", np.around(arm4j.get_endpoint_pos(),decimals=2))
            print("setting position to ", np.around(pos_list[self.count], decimals=2))
            arm4j.set_endpoint_pos(pos_list[self.count])
            time.sleep(2)
            print("final position = ", np.around(arm4j.get_endpoint_pos(),decimals=2))
            
            self.count+=1
        self.draw_world()

if __name__ == "__main__":
    win = Test2(window_size=(1920, 1080))
    #win.add_model(cone)
    win.add_model(arm4j)
    #win.screen_init()
    #win.draw_world()
    win.run()
