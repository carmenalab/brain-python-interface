from __future__ import division
import os
import threading
import operator

import pygame
import numpy as np
from OpenGL.GL import *

from riglib.experiment import LogExperiment, traits

from render import Renderer, SSAOrender
from models import Group
from utils import offaxis_frusta

class Window(LogExperiment):
    status = dict(draw=dict(stop=None))
    state = "draw"
    stop = False

    #window_size = (3840, 1080)
    window_size = (960, 270)
    background = (0,0,0,1)
    fps = 60

    #Screen parameters, all in centimeters -- adjust for monkey
    screen_dist = 35
    iod = 2.5
    fov = 45

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(**kwargs)
        self.models = []

    def init(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = "1680,0"
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"
        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS,1)
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
        
        self.renderer = SSAOrender(
            shaders=dict(
                passthru=(GL_VERTEX_SHADER, "passthrough.v.glsl"),
                default=(GL_FRAGMENT_SHADER, "default.f.glsl", "phong.f.glsl")),
            programs=dict(
                default=("passthru", "default"),
            ),
            win_size=self.window_size
        )
        
        w, h = self.window_size
        self.projections = offaxis_frusta((w/2,h), self.fov, 1, 1024, self.screen_dist, self.iod)
        
        #this effectively determines the modelview matrix
        self.world = Group(self.models)
        #up vector is always (0,0,1), why would I ever need to roll the camera?!
        self.set_eye((0,-self.screen_dist,0), (0,0))
        #Need to add extra Group to translate the eyes without affecting the modelview
        self.root = Group([self.world])
        self.root.init()
    
    def set_eye(self, pos, vec, reset=True):
        '''Set the eye's position and direction. Camera starts at (0,0,0), pointing towards positive y'''
        self.world.translate(-pos[0], -pos[1], -pos[2], reset=True).rotate_x(-90)
        self.world.rotate_x(vec[1]).rotate_y(vec[0])

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
        for x, side, projection in zip([0, int(w/2)], [.5,-.5], self.projections):
            glViewport(x, 0, int(w/2), h)
            self.root.translate(side*self.iod, 0, 0, reset=True)
            mv = np.dot(self.root.xfm, self.root.models[0].xfm)
            self.renderer.draw(self.root, p_matrix=projection, modelview=mv)
        
        pygame.display.flip()
        self.clock.tick()
        self.event = self._get_event()
    
    def _start_None(self):
        pygame.display.quit()
    
    def _test_stop(self, ts):
        return self.stop or self.event is not None and self.event[0] == 27
    
class FPScontrol(Window):
    '''A mixin that adds a WASD + Mouse controller to the window. 
    Use WASD to move in XY plane, q to go down, e to go up'''

    def init(self):
        super(FPScontrol, self).init()
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        self.eyepos = [0,-self.screen_dist, 0]
        self.eyevec = [0,0]
        self.wasd = [False, False, False, False, False, False]

    def _get_event(self):
        retme = None
        for e in pygame.event.get([pygame.MOUSEMOTION, pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT]):
            moved = any(self.wasd)
            if e.type == pygame.MOUSEMOTION:
                self.eyevec[0] += 0.5*e.rel[0]
                self.eyevec[1] += 0.5*e.rel[1]
                moved = True
            elif e.type == pygame.KEYDOWN:
                kn = pygame.key.name(e.key)
                if kn in ["escape", "q"]:
                    self.stop = True
                retme = (e.key, e.type)
            elif e.type == pygame.QUIT:
                self.stop = True

            if moved:
                self.set_eye(self.eyepos, self.eyevec, reset=True)
        return retme

class Anaglyph(Window):
    def __init__(self, window_size=None, **kwargs):
        super(Anaglyph, self).__init__(**kwargs)
        self.window_size = window_size
        self.iod = 2.5

    def init(self):
        pygame.init()
        if self.window_size is None:
            info = pygame.display.Info()
            self.window_size = info.current_w, info.current_h
        super(Anaglyph, self).init()
        w, h = self.window_size
        self.projections = offaxis_frusta((w,h), self.fov, 1, 1024, self.screen_dist, self.iod)
        #self.renderer.add_shader("anaphong", GL_FRAGMENT_SHADER, "phong_anaglyph.f.glsl")
        #replace old phong shader with anaphong
        #self.renderer.add_program("default", ("passthru", "anaphong"))
        #self.renderer.make_frametex("color", self.window_size)
    
    def _while_draw(self):
        w, h = self.window_size
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glClear(GL_COLOR_BUFFER_BIT)
        glClear(GL_DEPTH_BUFFER_BIT)
        glViewport(0,0,w,h)

        glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE)
        self.root.translate(0.5*self.iod, 0, 0, reset=True)
        mv = np.dot(self.root.xfm, self.root.models[0].xfm)
        self.renderer.draw(self.root, p_matrix=self.projections[0], modelview=mv)

        glClear(GL_DEPTH_BUFFER_BIT)
        glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE)
        self.root.translate(-0.5*self.iod, 0, 0, reset=True)
        mv = np.dot(self.root.xfm, self.root.models[0].xfm)
        self.renderer.draw(self.root, p_matrix=self.projections[1], modelview=mv)
        
        pygame.display.flip()
        self.clock.tick()
        self.event = self._get_event()