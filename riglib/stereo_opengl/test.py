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

def elbowpos(target, upperarm=20, lowerarm=20):
    m = upperarm
    n = lowerarm
    x, y, z = target

    a = (m**2*x**2+y*np.sqrt(-x**2*(m**4-2*m**2*n**2-2*m**2*x**2-2*m**2*y**2+n**4-2*n**2*x**2-2*n**2*y**2+x**4+2*x**2*y**2+x**2*z**2+y**4+y**2*z**2))-n**2*x**2+x**4+x**2*y**2)/(2*x*(x**2+y**2))
    b = (m**2*y-np.sqrt(-x**2*(m**4-2*m**2*n**2-2*m**2*x**2-2*m**2*y**2+n**4-2*n**2*x**2-2*n**2*y**2+x**4+2*x**2*y**2+x**2*z**2+y**4+y**2*z**2))-n**2*y+x**2*y+y**3)/(2*(x**2+y**2))

    return a, b, z/2

def rotations(target, elbow):
    '''Assumes the initial position will be (0,0,0) and initial rotation to be 
    (0,0,1) on both upperarm and forearm'''
    target, elbow = np.array(target), np.array(elbow)
    r1, r2 = [0,0,0], [0,0,0]
    elbow /= np.sqrt((elbow**2).sum())
    r1[1] = np.arccos(elbow[2])*np.sign(elbow[2])
    r1[2] = np.arcsin(elbow[1] / np.arcsin(elbow[2]))
    print elbow[1] / np.arcsin(elbow[2])
    return r1, r2


FlatSphere = type("FlatSphere", (Sphere, FlatMesh), globals())
TexPlane = type("TexPlane", (Plane, TexModel), globals())
bounce = FlatSphere(radius=8, color=(0.6,0.2,0.2,1), shininess=50).translate(10,20,-15)

forearm = Group([Cylinder(radius=2.5, height=20).rotate_y(90), Sphere(3).translate(20, 0, 0)]).translate(-20, 0, 0).rotate_y(90)
upperarm = Group([
    Cylinder(radius=2.5, height=20).rotate_y(-90), 
    Sphere(3).translate(-20, 0, 0), 
    forearm
    ]).rotate_y(-30)
arm = Group([upperarm])

class Test(FPScontrol, Window):
    def _get_renderer(self):
        mirrorSSAO = type("mirrorSSAO", (stereo.RightLeft, ssao.SSAO), globals())
        return mirrorSSAO(self.window_size, self.fov, 1., 1024., self.screen_dist, self.iod)

    def _while_draw(self):
        ts = time.time() - self.start_time
        arm.rotate_x((ts/5.)*360, reset=True)
        forearm.rotate_y((ts/3.)*360, reset=True)
        super(Test, self)._while_draw()
        if int(ts) % 5 == 0:
            print self.clock.get_fps()
        self.renderer.draw_done()

if __name__ == "__main__":
    win = Test(window_size=(1366, 400))
    tex = cloudy_tex((1024, 1024))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, -15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).rotate_x(90).translate(-250, 250,-15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, 0).rotate_y(-90).translate(250, 0, 250-15))
    win.add_model(TexPlane(500,500, tex=tex, specular_color=(0.,0,0,0)).translate(-250, -250, 0).rotate_y(90).translate(-250, 0, 250-15))
    win.add_model(Sphere(radius=4, color=(0., 0.4, 0., 1), shininess=30).translate(-20, 10, -11))
    win.add_model(bounce)
    win.add_model(arm)
    win.run()
