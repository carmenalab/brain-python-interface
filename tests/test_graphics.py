import sys

import time
import numpy as np

import os

from riglib.pupillabs import utils
from riglib.stereo_opengl.textures import Texture
os.environ['DISPLAY'] = ':0'

from riglib.stereo_opengl.environment import Grid
from riglib.stereo_opengl.window import Window, Window2D, FPScontrol
from riglib.stereo_opengl.primitives import AprilTag, Cylinder, Cube, Plane, Snake, Sphere, Cone, Text, Cable, TexSphere, TexCube, TexPlane
from features.optitrack_features import SpheresToCylinders
from riglib.stereo_opengl.window import Window, Window2D, FPScontrol, WindowSSAO, Window2DIn3D
from riglib.stereo_opengl.openxr import WindowVR
from riglib.stereo_opengl.environment import Box, Grid
from riglib.stereo_opengl.primitives import Cylinder, Cube, Plane, Sphere, Cone, Text, TexSphere, TexCube, TexPlane, AprilTag, CalibrationSphere
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.render import ssao, stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex

from riglib.stereo_opengl.ik import RobotArmGen2D
from riglib.stereo_opengl.xfm import Quaternion, Transform
import time

from riglib.stereo_opengl.ik import RobotArm

from built_in_tasks.target_graphics import TextTarget, target_colors
from built_in_tasks.target_capture_task import ScreenTargetCapture
import pygame
from OpenGL.GL import *

# arm4j = RobotArmGen2D(link_radii=.2, joint_radii=.2, link_lengths=[4,4,2,2])
travel_speed = 0.5
travel_radius = 10
moon = Sphere(radius=0.5, color=[0.25,0.25,0.75,0.5])
planet = Sphere(3, color=[0.75,0.25,0.25,0.75])
orbit_radius = 4
orbit_speed = 1
wobble_radius = 5
wobble_speed = 0.5
#TexSphere = type("TexSphere", (Sphere, TexModel), {})
#TexPlane = type("TexPlane", (Plane, TexModel), {})
reward_text = Text(7.5, "123", justify='right', color=[1,0,1,1])
# center_out_gen = ScreenTargetCapture.centerout_2D(1)
# center_out_positions = [pos[1] for _, pos in center_out_gen]
cable = Snake(0.5, 2*np.sin(np.arange(200)/2), color=(1,0,1,0.75)).translate(-15, 0, -10)
center_out_gen = ScreenTargetCapture.centerout_tabletop(1)
center_out_positions = [(pos[1][0], pos[1][1], -10) for _, pos in center_out_gen]
center_out_targets = [
    Sphere(radius=2, color=target_colors['yellow']).translate(*pos)
    for pos in center_out_positions
]
center_out_targets[6].color = target_colors['cyan']
apriltag = AprilTag(0, size=4)
apriltag.translate(0, 0, -10)
moon = CalibrationSphere(radius=2)

pos_list = np.array([[0,0,0],[0,0,5]])

class Test2(Window2DIn3D, Window):

    def __init__(self, *args, **kwargs):
        self.count=0
        self.cursor_bounds = np.array([-15, 15, -15, 15, -15, 15])
        super().__init__(*args, **kwargs)

    def _start_draw(self):
        #arm4j.set_joint_pos([0,0,np.pi/2,np.pi/2])
        #arm4j.get_endpoint_pos()
        self.add_environment_model(Grid(50))
        self.add_model(moon)
        self.add_model(planet)
        self.add_model(apriltag)
        # self.add_model(moon)
        # self.add_model(planet)
        # self.add_model(arm4j)
        #self.add_model(reward_text.translate(5,0,-5))
        # self.add_model(TexSphere(radius=3, specular_color=[1,1,1,1], tex=cloudy_tex()).translate(5,0,0))
        # self.add_model(TexPlane(5,5, tex=cloudy_tex(), specular_color=(0.,0,0,1)).rotate_x(90))
        # self.add_model(TexPlane(5,5, specular_color=(0.,0,0,1), tex=cloudy_tex()).rotate_x(90))
        # reward_text = Text(7.5, "123", justify='right', color=[1,0,1,0.75])
        self.add_model(reward_text)
        # self.add_model(TexPlane(4,4,color=[0,0,0,0.9], tex=cloudy_tex()).rotate_x(90).translate(0,0,-5))
        #self.screen_init()
        #self.draw_world()
        for model in center_out_targets:
            self.add_model(model)
        for model in center_out_targets:
            self.add_model(model)
        self.add_model(Sphere(radius=1, color=target_colors['purple']).translate(3,0,-10))
        self.add_model(cable)

    def _while_draw(self):
        ts = time.time() - self.start_time
        # cable.update_texture(int(ts*10), len(cable.trajectory))
        
        x = travel_radius * np.cos(ts * travel_speed)
        y = travel_radius * np.sin(ts * travel_speed)

        pos = np.array([x, 0, y])

        x = orbit_radius * np.cos(ts * orbit_speed)
        z = orbit_radius * np.sin(ts * orbit_speed)

        moon.translate(x+pos[0],z+pos[1],pos[2],reset=True)
        moon.look_at([0,-40,0])
        planet.translate(pos[0], pos[1], pos[2],reset=True)

        x = wobble_radius * np.cos(ts * wobble_speed)
        y = wobble_radius * np.sin(ts * wobble_speed)

        xfm = Transform(move=[x,0,-self.screen_dist])
        xfm.rotate_x(np.radians(y))
        xfm.rotate_y(np.radians(x))
        self.modelview = xfm.to_mat()

        if ts > 2 and self.count<len(pos_list):
            # reward_text.translate(*pos_list[self.count])
            self.count+=1
        if ts > 4 and self.count<len(pos_list)+1:
            # win.remove_model(reward_text)
            # target = TextTarget('hi', [1,1,0,1], 1)
            # win.add_model(target.model)
            self.count += 1
        self.draw_world()

        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL error after drawing: {error}")


if __name__ == "__main__":
    win = Test2(window_size=(1000, 800), fullscreen=False, stereo_mode='projection',
                screen_dist=50, screen_half_height=22.5, 
                background=(0.5,0.5,0.5,1),
                overlay_plane_pos=(0,-10,0),
                overlay_plane_rot_x=0)
    win.run()
