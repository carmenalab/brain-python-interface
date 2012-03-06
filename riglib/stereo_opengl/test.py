from __future__ import division

import time
import numpy as np
import pygame

from window import Window, FPScontrol
from primitives import Cylinder, Plane, Sphere
from models import FlatMesh, Group
from textures import Texture, TexModel
from render import ssao, stereo, Renderer
from utils import cloudy_tex
from ik import TwoJoint


FlatSphere = type("FlatSphere", (Sphere, FlatMesh), globals())
TexPlane = type("TexPlane", (Plane, TexModel), globals())
bounce = FlatSphere(radius=8, color=(0.6,0.2,0.2,1), shininess=50).translate(10,20,-15)

forearm = Group([Cylinder(radius=2.5, height=20), Sphere(3).translate(20, 0, 0)]).translate(0,0,20)
upperarm = Group([
    Cylinder(radius=2.5, height=20), Sphere(3).translate(0, 0, 20), 
    forearm])

system = TwoJoint(upperarm, forearm)

class Test(Window):
    def _get_renderer(self):
        mirrorSSAO = type("mirrorSSAO", (stereo.RightLeft, ssao.SSAO), globals())
        return stereo.RightLeft(self.window_size, self.fov, 1., 1024., self.screen_dist, self.iod)

    def _while_draw(self):
        ts = time.time() - self.start_time
        t = (ts/5.)*2*np.pi
        #system.set((np.cos(t),1,np.sin(t)))

        super(Test, self)._while_draw()
        if int(ts) % 5 == 0:
            print self.clock.get_fps()
        self.renderer.draw_done()

if __name__ == "__main__":
    win = Test(window_size=(1920,540))
    tex = cloudy_tex((1024, 1024))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, -15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_x(90).translate(-250, 250,-15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, 0).rotate_y(-90).translate(250, 0, 250-15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, 0).rotate_y(90).translate(-250, 0, 250-15))
    win.add_model(Sphere(radius=4, color=(0., 0.4, 0., 1), shininess=30).translate(0,0,0))
#    win.add_model(bounce)
#    win.add_model(upperarm)
    win.run()
