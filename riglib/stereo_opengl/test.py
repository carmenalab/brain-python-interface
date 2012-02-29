import time
import numpy as np
import pygame

from window import Window, FPScontrol
from primitives import Cylinder, Plane, Sphere
from models import FlatMesh, Group, Texture, TexModel
from render import ssao
from utils import cloudy_tex

FlatSphere = type("FlatSphere", (Sphere, FlatMesh), globals())
TexPlane = type("TexPlane", (Plane, TexModel), globals())
sphere = FlatSphere(radius=8, color=(0.6,0.4,0.4,1), shininess=50).translate(10,20,-4)

fc = Cylinder(height=20, segments=20, color=(0.3, 0.3, 0.6,1), specular_color=(0.,0,0,0))
fcg = Group([fc]).rotate_y(90)

forearm = Group([Cylinder(radius=2.5, height=20).rotate_y(90), Sphere(3).translate(-20,0,0)])

class Test(Window):
    def _get_renderer(self):
        return ssao.SSAO(self.window_size, self.fov, 1., 1024.)

    def _while_draw(self):
        ts = time.time() - self.start_time
        fc.rotate_x((ts/5.)*360, reset=True)
        super(Test, self)._while_draw()

if __name__ == "__main__":
    win = Test(window_size=(1280, 640))
    tex = cloudy_tex()
    win.add_model(TexPlane(200,100, tex=tex).translate(-100, 0, -15))
    win.add_model(Sphere(radius=4, color=(0., 0.4, 0., 1), shininess=30).translate(-20, 20, -11))
    win.add_model(sphere)
    win.add_model(fcg)
    win.run()
