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

#from riglib import source, motiontracker
#Motion = motiontracker.make(8, motiontracker.System)
#sys = source.DataSource(Motion)	
#sys.start()

FlatSphere = type("FlatSphere", (Sphere, FlatMesh), {})
TexPlane = type("TexPlane", (Plane, TexModel), {})
TexSphere = type("TexSphere", (Sphere, TexModel), {})
tex = cloudy_tex((1024, 1024))

arm = RobotArm()
arm2 = RobotArm2D()
arm_color = (1,1,1,1) # Color and transparency of arm
arm_radius = .3 # Radius of arm links
arm_link_lengths = [15.,20.] # Length of lower, upper arm
arm3 = RobotArm2J2D(link_radii = [arm_radius, arm_radius], joint_radii = [arm_radius, arm_radius],
        link_lengths=arm_link_lengths, joint_colors = [arm_color, arm_color], link_colors = [arm_color, arm_color])
ball = Sphere(radius=5, color=(0.5, 1, 0.5, 1), shininess=20)

class Test2(Window):
    background = (0,0,0,1)
    def _get_renderer(self):
        mirrorSSAO = type("mirrorSSAO", (stereo.MirrorDisplay, ssao.SSAO), globals())
        return mirrorSSAO(self.window_size, self.fov, 1., 1024., self.screen_dist, self.iod)

    def _start_draw(self):
        #arm3.set_joint_pos([0.0, 0.0])
        arm3.set_endpoint_pos((0,0.,5.))
        #print arm3.curr_vecs
        #p = arm3.get_joint_pos()
        #print np.round(p[0],decimals=2), np.round(p[1],decimals=2)#, np.round(p[2],decimals=2)
        #pass

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
    win.run()