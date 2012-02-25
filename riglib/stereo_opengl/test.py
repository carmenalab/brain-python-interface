import time
import numpy as np
import pygame

from window import Window
from primitives import Cylinder, Plane
from models import FlatMesh, Group

FlatCyl = type("FlatCyl", (Cylinder, FlatMesh), globals())
fc = FlatCyl(height=10, segments=12, color=(0.3, 0.3, 0.6,1))
fcg = Group([fc]).translate(-2,10,0).rotate_y(90)

class Test(Window):
    def _while_draw(self):
        #ts = time.time() - self.start_time
        #fc.rotate_x((ts/10.)*360, reset=True)
        super(Test, self)._while_draw()
        if self.event is not None:
            kn = pygame.key.name(self.event[0])
            if kn == "left":
                fcg.rotate_x(5)
            elif kn == "right":
                fcg.rotate_x(-5)


if __name__ == "__main__":
    win = Test()
    win.add_model(Cylinder(height=2, segments=128, color=(0.6,0.4,0.4,1)).translate(1,2,-1))
    win.add_model(fcg)
    win.run()