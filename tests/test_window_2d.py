#!/usr/bin/python
from riglib.stereo_opengl import window
from riglib.stereo_opengl.primitives import Sphere
import pygame

reload(window)

target_radius = 2.
target = Sphere(radius=target_radius, color=(1,0,0,.5))
target.xfm.translate(0, 0, 0, reset=True)

class SimWindow(window.Window2D):
    def _while_draw(self):
        print "while draw"
        target.attach()
        print target.radius
        self.requeue()
        self.draw_world()

w = SimWindow()
w.add_model(target)
w.run()
