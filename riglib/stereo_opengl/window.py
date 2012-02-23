import os
import threading
import operator

import pygame
import numpy as np
from OpenGL.GL import *

from riglib.experiment import LogExperiment, traits

from world import Context, perspective
from models import Group

cwd = os.path.abspath(os.path.split(__file__)[0])

class Window(LogExperiment):
    status = dict(draw=dict(stop=None))
    state = "draw"
    stop = False

    background = (0,0,0,1)
    fps = 60
    fov = 90

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__()
        self.models = []
        self.eyepos = [0,-2,0]

    def init(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "1680,0"
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"
        pygame.init()

        flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.OPENGL
        pygame.display.set_mode((800,600), flags)
        self.clock = pygame.time.Clock()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*self.background)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glViewport(0,0,800,600)

        self.ctx = Context(open(os.path.join(cwd, "test.v.glsl")), open(os.path.join(cwd, "test.f.glsl")))
        self.projection = perspective(self.fov/2, 800./600, 0.0625, 256)
        #this effectively determines the modelview matrix
        self.world = Group(self.models).rotate_x(-90).translate(*map(operator.neg, self.eyepos))
        self.world.init()

    def add_model(self, model):
        self.models.append(model)
    
    def run(self):
        self.init()
        return super(Window, self).run()
    
    def _get_event(self):
        for e in pygame.event.get(pygame.KEYDOWN):
            return (e.key, e.type)
    
    def _while_draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.ctx.program)
        self.ctx.uniforms['p_matrix'] = self.projection
        self.world.draw(self.ctx)
        pygame.display.flip()
        self.clock.tick(self.fps)
        self.event = self._get_event()
    
    def _test_stop(self, ts):
        return self.stop or self.event is not None and self.event[0] == 27