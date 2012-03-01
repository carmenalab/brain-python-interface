import time
import numpy as np
import pygame

from window import Window, FPScontrol
from primitives import Cylinder, Plane, Sphere
from models import FlatMesh, Group, Texture, TexModel
from render import ssao, stereo, Renderer
from utils import cloudy_tex

FlatSphere = type("FlatSphere", (Sphere, FlatMesh), globals())
TexPlane = type("TexPlane", (Plane, TexModel), globals())
sphere = FlatSphere(radius=8, color=(0.6,0.2,0.2,1), shininess=50).translate(10,20,-4)

fc = Cylinder(height=20, segments=20, color=(0.3, 0.3, 0.6,1), shininess=20).rotate_x(60)
fcg = Group([fc]).rotate_y(90)

forearm = Group([Cylinder(radius=2.5, height=20).rotate_y(90), Sphere(3).translate(-20,0,0)])

class Test(Window):
    def _get_renderer(self):
        mirrorSSAO = type("mirroSSAO", (stereo.MirrorDisplay, ssao.SSAO), globals())
        return mirrorSSAO(self.window_size, self.fov, 1., 512., self.screen_dist, self.iod)

    def _while_draw(self):
        ts = time.time() - self.start_time
        fc.rotate_x((ts/5.)*360, reset=True)
        super(Test, self)._while_draw()
        if int(ts) % 5 == 0:
            print self.clock.get_fps()
        self.renderer.draw_done()

if __name__ == "__main__":
    win = Test()
    tex = cloudy_tex((1024, 1024))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, -15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_x( 90).translate(-250,  250,-15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_y(-90).translate( 250, -250,-15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_y(-90).translate(-250, -250,-15))
    win.add_model(Sphere(radius=4, color=(0., 0.4, 0., 1), shininess=30).translate(-20, 10, -11))
    win.add_model(sphere)
    win.add_model(fcg)
    win.run()
