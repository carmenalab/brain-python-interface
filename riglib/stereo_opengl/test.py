import time
import numpy as np
import pygame

from window import Window, Anaglyph, FPScontrol
from primitives import Cylinder, Plane, Sphere
from models import FlatMesh, Group, Texture, TexModel
from utils import cloudy_tex

FlatSphere = type("FlatSphere", (Sphere, FlatMesh), globals())
TexPlane = type("TexPlane", (Plane, TexModel), globals())
sphere = FlatSphere(radius=8, color=(0.6,0.4,0.4,1), shininess=50).translate(10,20,-4)

fc = Cylinder(height=20, segments=20, color=(0.3, 0.3, 0.6,1), specular_color=(0.,0,0,0))
fcg = Group([fc]).rotate_y(90)

class Test(Anaglyph):
    def _while_draw(self):
        ts = time.time() - self.start_time
        fc.rotate_x((ts/10.)*360, reset=True)
        super(Test, self)._while_draw()

if __name__ == "__main__":
    win = Test()
    tex = Texture(cloudy_tex((2048,2048)))
    win.add_model(TexPlane(200,100, tex=tex).translate(-100, 0, -30))
    win.add_model(Sphere(radius=4, color=(0., 0.4, 0., 1), shininess=30).translate(-20, 10, 0))
    win.add_model(sphere)
    win.add_model(fcg)
    win.start()
