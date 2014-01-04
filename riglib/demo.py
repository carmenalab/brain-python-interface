from __future__ import division
import sys

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

from riglib.stereo_opengl.ik import RobotArm

import pygame


#from riglib import source, motiontracker
#Motion = motiontracker.make(8, motiontracker.System)
#sys = source.DataSource(Motion)	
#sys.start()


#FlatSphere = type("FlatSphere", (Sphere, FlatMesh), {})
#TexPlane = type("TexPlane", (Plane, TexModel), {})
#TexSphere = type("TexSphere", (Sphere, TexModel), {})
#tex = cloudy_tex((1024, 1024))

#arm = RobotArm()
#ball = Sphere(radius=5, color=(0.5, 1, 0.5, 1), shininess=20)

class Test(Window):
    ##def _get_renderer(self):
    ##    mirrorSSAO = type("mirrorSSAO", (stereo.MirrorDisplay, ssao.SSAO), globals())
    ##    return mirrorSSAO(self.window_size, self.fov, 1., 1024., self.screen_dist, self.iod)

    def _while_draw(self):
        print 'while_draw'
        pts = sys.get().mean(0)[...,:-1]+[-20,0,-5]
        arm.set(pts[4])
        ts = time.time() - self.start_time
        t = (ts / 2.) * 2*np.pi
        ball.translate(0,100,20*np.abs(np.sin(t))-10, reset=True)
        #t2 = (ts / 2.) * 2*np.pi
        #arm.set((np.cos(t)*10-10, np.sin(t2)*10+20, np.sin(t)*15))        
        self.draw_world()
        #print self.clock.get_fps()

class Test2(Window):
    ##def _get_renderer(self):
    ##    mirrorSSAO = type("mirrorSSAO", (stereo.MirrorDisplay, ssao.SSAO), globals())
    ##    return mirrorSSAO(self.window_size, self.fov, 1., 1024., self.screen_dist, self.iod)

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
        ts = time.time() - self.start_time
        t = (ts / 2.) * np.pi
#        arm.set_joints_2D(0.0, t)
        #ball.translate(0,100,20*np.abs(np.sin(t))-10, reset=True)       
        self.draw_world()

##if __name__ == "__main__":
##    win = Test(window_size=(1920*2, 1080))
##    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, -15))
##    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_x(90).translate(-250, 250,-15))
##    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_y(-90).translate(250,-250,-15))
##    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_y(90).translate(-250,-250,-15))
##    win.add_model(TexSphere(radius=4, shininess=30, tex=tex).translate(-20, 10, -11))
##    win.add_model(FlatSphere(radius=8, color=(0.6,0.2,0.2,1), shininess=50).translate(10,20,-15))
##    win.add_model(arm)
##    win.add_model(ball.translate(0, 100, -10))
##    win.run()
##    sys.stop()

if __name__ == "__main__":
    win = Test2(window_size=(1920, 1080))
    win.add_model(Plane(50, 50, color=(.6, .2, .2, 1)).rotate_x(90).translate(-25, 0, -25))
    #win.add_model(arm)
    #win.screen_init()
    #win.draw_world()
    win.run()
#    import time
#    try:
#        while True:
#            time.sleep(.1)
#    except KeyboardInterrupt:
#        pass