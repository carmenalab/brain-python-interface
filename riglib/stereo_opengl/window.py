from __future__ import division
import os
import threading
import operator

import pygame
import numpy as np
from OpenGL.GL import *

from riglib.experiment import LogExperiment, traits

from world import World
from models import Group
from utils import perspective

class Window(LogExperiment):
    status = dict(draw=dict(stop=None))
    state = "draw"
    stop = False
    window_size = (960, 270)
    iod = 2.5

    background = (0,0,0,1)
    fps = 60
    fov = 60

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__()
        self.models = []
        self.eyepos = [0,-2,0]

    def init(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "1680,0"
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.OPENGL
        pygame.display.set_mode(self.window_size, flags)
        self.clock = pygame.time.Clock()

        glEnable(GL_BLEND)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_TEXTURE_2D)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*self.background)
        glClearDepth(1.0)
        glDepthMask(GL_TRUE)
        
        self.programs = Programs(
            shaders=dict(
                passthru=(GL_VERTEX_SHADER, "passthrough.v.glsl"),
                #flat_geom=(GL_GEOMETRY_SHADER, "flat_shade.g.glsl"),
                #smooth_geom=(GL_GEOMETRY_SHADER, "smooth_shade.g.glsl"),
                phong=(GL_FRAGMENT_SHADER, "phong.f.glsl")), 
            programs=dict(
                #flat=("passthru","flat_geom","phong"),
                default=("passthru", "phong"),
                #smooth=("passthru", "smooth_geom", "phong"),
            )
        )
        
        w, h = self.window_size
        self.projection = perspective(self.fov/2, (w/2)/h, 0.0625, 256.)

        #this effectively determines the modelview matrix
        world = Group(self.models).rotate_x(-90).translate(*map(operator.neg, self.eyepos))
        
        self.group = 

    def add_model(self, model):
        self.models.append(model)
    
    def run(self):
        self.init()
        return super(Window, self).run()
    
    def _get_event(self):
        for e in pygame.event.get(pygame.KEYDOWN):
            return (e.key, e.type)
    
    def _while_draw(self):
        w, h = self.window_size
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for side, projection in enumerate(['left', 'right']):
            glViewport((0, int(w/2))[side], 0, int(w/2), h)
            self.world.draw(self.root, self.projection, self.root.xfm)
        
        pygame.display.flip()
        self.clock.tick()
        self.event = self._get_event()
    
    def _start_None(self):
        pygame.display.quit()
    
    def _test_stop(self, ts):
        return self.stop or self.event is not None and self.event[0] == 27