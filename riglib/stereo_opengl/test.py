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
from ik import RobotArm

from riglib import shm

#sys = shm.MemTrack()
#sys.start("motion")

FlatSphere = type("FlatSphere", (Sphere, FlatMesh), {})
TexPlane = type("TexPlane", (Plane, TexModel), {})
TexSphere = type("TexSphere", (Sphere, TexModel), {})
tex = cloudy_tex((1024, 1024))

arm = RobotArm()

class Test(Window):
    def _get_renderer(self):
        mirrorSSAO = type("mirrorSSAO", (stereo.MirrorDisplay, ssao.SSAO), globals())
        return stereo.Anaglyph(self.window_size, self.fov, 1., 1024., self.screen_dist, self.iod)

    def _while_draw(self):
        #pts = sys.get("motion").reshape(-1, 8, 3)[:,6].mean(0)
        #arm.set(pts[6]*0.1)
        ts = time.time() - self.start_time
        t = (ts / 5.) * 2*np.pi
        t2 = (ts / 2.) * 2*np.pi
        arm.set((np.cos(t)*10-10, np.sin(t2)*10+20, np.sin(t)*15))
        super(Test, self)._while_draw()
        
        self.renderer.draw_done()

if __name__ == "__main__":
    win = Test(window_size=(1920,1200))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, -15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_x(90).translate(-250, 250,-15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_y(-90).translate(250,-250,-15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_y(90).translate(-250,-250,-15))
    win.add_model(TexSphere(radius=4, shininess=30, tex=tex).translate(-20, 10, -11))
    win.add_model(FlatSphere(radius=8, color=(0.6,0.2,0.2,1), shininess=50).translate(10,20,-15))
    win.add_model(arm.translate(12,-20,0))
    win.run()
    sys.stopall()
