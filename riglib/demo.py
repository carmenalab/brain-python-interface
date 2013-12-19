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
arm3 = RobotArm2J2D(link_lengths = [10, 5])
ball = Sphere(radius=5, color=(0.5, 1, 0.5, 1), shininess=20)

# class Test(Window):
#     def _get_renderer(self):
#         mirrorSSAO = type("mirrorSSAO", (stereo.MirrorDisplay, ssao.SSAO), globals())
#         return mirrorSSAO(self.window_size, self.fov, 1., 1024., self.screen_dist, self.iod)

#     def _while_draw(self):
#         pts = sys.get().mean(0)[...,:-1]+[-20,0,-5]
#         arm.set(pts[4])
#         ts = time.time() - self.start_time
#         t = (ts / 2.) * 2*np.pi
#         ball.translate(0,100,20*np.abs(np.sin(t))-10, reset=True)
#         #t2 = (ts / 2.) * 2*np.pi
#         #arm.set((np.cos(t)*10-10, np.sin(t2)*10+20, np.sin(t)*15))        
#         self.draw_world()
#         #print self.clock.get_fps()

class Test2(Window):
    background = (.1,.1,.1,0)
    def _get_renderer(self):
        mirrorSSAO = type("mirrorSSAO", (stereo.MirrorDisplay, ssao.SSAO), globals())
        return mirrorSSAO(self.window_size, self.fov, 1., 1024., self.screen_dist, self.iod)

    def _start_draw(self):
        arm3.set_joint_pos([0.0, 0.0])

    def _while_draw(self):
        ts = time.time() - self.start_time
        t = (ts / 2.) * np.pi
        #arm.set_joints_2D(t, 0.0)
        #arm2.set_joint_pos([t])
        #arm2.set_endpoint_pos(5*np.cos(t),0,5*np.sin(t))
        arm3.set_joint_pos([t, t])
        #print t, arm2.get_joint_pos()
        #ball.translate(0,100,20*np.abs(np.sin(t))-10, reset=True)
        #arm2.set_joint_pos([0.0])
        #arm2.set_joint_pos([np.pi/2])
        #arm2.set_joint_pos([np.pi]) 
        self.draw_world()

# if __name__ == "__main__":
#     win = Test(window_size=(1920*2, 1080))
#     win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, -15))
#     win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_x(90).translate(-250, 250,-15))
#     win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_y(-90).translate(250,-250,-15))
#     win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_y(90).translate(-250,-250,-15))
#     win.add_model(TexSphere(radius=4, shininess=30, tex=tex).translate(-20, 10, -11))
#     win.add_model(FlatSphere(radius=8, color=(0.6,0.2,0.2,1), shininess=50).translate(10,20,-15))
#     win.add_model(arm)
#     win.add_model(ball.translate(0, 100, -10))
#     win.run()
#     sys.stop()

if __name__ == "__main__":
    win = Test2(window_size=(1920*2, 1080))
    #win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_x(90).translate(-250, 250,-15))
    #win.add_model(arm2)
    #win.add_model(arm)
    win.add_model(arm3)
    win.run()