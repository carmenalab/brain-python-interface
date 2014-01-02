from __future__ import division

import time
import numpy as np

from riglib.stereo_opengl.window import Window, FPScontrol
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import ssao, stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex
from riglib.stereo_opengl.ik import RobotArm, RobotArm2D, RobotArm2J2D
from stereo_opengl.xfm import Quaternion
import time
#from riglib import source, motiontracker
#Motion = motiontracker.make(8, motiontracker.System)
#sys = source.DataSource(Motion)	
#sys.start()

import os
os.environ['DISPLAY'] = ':0'

FlatSphere = type("FlatSphere", (Sphere, FlatMesh), {})
TexPlane = type("TexPlane", (Plane, TexModel), {})
TexSphere = type("TexSphere", (Sphere, TexModel), {})
tex = cloudy_tex((1024, 1024))

arm = RobotArm()
arm2 = RobotArm2D()
arm_color = (1,1,1,1) # Color and transparency of arm
arm_color2 = (.5,1,1,1)
arm_radius = .3 # Radius of arm links
arm_link_lengths = [5,5.] # Length of lower, upper arm
arm3 = RobotArm2J2D(link_radii = [arm_radius, arm_radius], joint_radii = [arm_radius, arm_radius],
        link_lengths=arm_link_lengths, joint_colors = [arm_color, arm_color], link_colors = [arm_color, arm_color2])
ball = Sphere(radius=5, color=(0.5, 1, 0.5, 1), shininess=20)

stick = Group([Cylinder(radius=arm_radius, height=arm_link_lengths[1], color=arm_color)])
pi = np.pi
class Test2(Window):
    background = (.5,.5,.5,1)
    def _get_renderer(self):
        mirrorSSAO = type("mirrorSSAO", (stereo.MirrorDisplay, ssao.SSAO), globals())
        return mirrorSSAO(self.window_size, self.fov, 1., 1024., self.screen_dist, self.iod)

    def _start_draw(self):
        #arm3.set_joint_pos([pi/2, pi/2])
        #arm3.set_endpoint_pos([3, 0, 6])
        #print "endpoint = ", arm3.get_endpoint_pos()
        #print 'joint pos (deg)', arm3.get_joint_pos() * 180/np.pi
        pos = np.array([5,0,5])
        print "starting position = ",pos 
        angs = arm3.perform_ik(pos)
        angs2 = np.array([angs['el_pflex'][0], angs['sh_pabd'][0]])
        print "angles = ", angs2
        pos2 = arm3.perform_fk(angs2)
        print "final position =", pos2
        arm3.set_endpoint_pos(pos2)

    def _while_draw(self):
        ts = (time.time() - self.start_time)/10.
        t = (ts) * np.pi
        #arm.set_joints_2D(t, 0.0)
        #arm2.set_joint_pos([t])
        #arm2.set_endpoint_pos(5*np.cos(t),0,5*np.sin(t))
        #arm3.set_joint_pos([t, t])
        #arm3.set_endpoint_pos(1+4*np.cos(t),0,4*np.sin(t)*np.cos(t))
        #print arm3.get_endpoint_pos()
        self.draw_world()

if __name__ == "__main__":
    win = Test2(window_size=(1920*2, 1080))
    #win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_x(90).translate(-250, 250,-15))
    #win.add_model(arm2)
    #win.add_model(arm)
    win.add_model(arm3)
    #win.add_model(stick)
    #win.add_model(Sphere(radius=5, color=(0,0,0,1)).translate(10,0,0))
    #win.add_model(Sphere(radius=5, color=(0,0,0,1), shininess=50))
    win.run()
