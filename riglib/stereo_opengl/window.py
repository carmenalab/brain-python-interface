from __future__ import division
import os
import threading
import operator

import pygame
import numpy as np
from OpenGL.GL import *

from riglib.experiment import LogExperiment, traits

from render import Renderer
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
    screen_dist = 40
    iod = 6.7
    fov = 45

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(**kwargs)
        self.models = []

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
        
        self.renderer = Renderer(
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
        self.projections = offaxis_frusta((w/2,h), self.fov, 0.1, 1024, self.screen_dist, self.iod)

        #this effectively determines the modelview matrix
        world = Group(self.models).translate(0, self.screen_dist, 0).rotate_x(-90)
        #Need to add extra Group to translate the eyes without affecting the modelview
        self.root = Group([world])
        self.root.init()

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

class Anaglyph(Window):
    def init(self):
        pygame.init()
        info = pygame.display.Info()
        self.window_size = info.current_w, info.current_h
        super(Anaglyph, self).init()
        w, h = self.window_size
        self.projections = offaxis_frusta((w,h), self.fov, 0.1, 1024, 40, self.iod)
        self.renderer = Renderer(
            shaders=dict(
                passthru=(GL_VERTEX_SHADER, "passthrough.v.glsl"),
                phong=(GL_FRAGMENT_SHADER, "phong_anaglyph.f.glsl")), 
            programs=dict(
                default=("passthru", "phong"),
            )
        )
        self.renderer.make_frametex("color", self.window_size)

    def _while_draw(self):
        def activate_tex(ctx):
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.renderer.frametexs['color'])
            ctx.uniforms['fbo'] = 0

        w, h = self.window_size
        glClear(GL_COLOR_BUFFER_BIT)
        glClear(GL_DEPTH_BUFFER_BIT)
        glViewport(0,0,w,h)
        self.root.translate(0.5*self.iod, 0, 0, reset=True)
        mv = np.dot(self.root.xfm, self.root.models[0].xfm)
        self.renderer.draw_to_fbo(self.root, p_matrix=self.projections[0], modelview=mv)
        glClear(GL_DEPTH_BUFFER_BIT)
        self.root.translate(0.5*self.iod, 0, 0, reset=True)
        mv = np.dot(self.root.xfm, self.root.models[0].xfm)
        self.renderer.draw(self.root, p_matrix=self.projections[1], modelview=mv, activate_tex=activate_tex)
        
        pygame.display.flip()
        self.clock.tick()
        self.event = self._get_event()