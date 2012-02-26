import time
import numpy as np
import pygame

from window import Window, Anaglyph
from primitives import Cylinder, Plane, Sphere
from models import FlatMesh, Group, Texture
from utils import cloudy_tex

FlatSphere = type("FlatSphere", (Sphere, FlatMesh), globals())
sphere = FlatSphere(radius=8, segments=16, color=(0.6,0.4,0.4,1), shininess=50).translate(10,-10,-4)

fc = Cylinder(height=20, segments=20, color=(0.3, 0.3, 0.6,1), shininess=10)
fcg = Group([fc]).rotate_y(90)

class Test(Anaglyph):
    def _while_draw(self):
        ts = time.time() - self.start_time
        #fc.rotate_x((ts/10.)*360, reset=True)
        super(Test, self)._while_draw()
        
        if self.event is not None:
            kn = pygame.key.name(self.event[0])
            if kn == "left":
                fc.rotate_x(5)
            elif kn == "right":
                fc.rotate_x(-5)
            


if __name__ == "__main__":
    tex = Texture(cloudy_tex())
    win = Test(window_size=(800,600))
    win.add_model(Plane(500,100, tex=tex).translate(-250, 0, -50))
    win.add_model(sphere)
    win.add_model(fcg)
    win.run()
