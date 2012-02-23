import threading

import pygame
import numpy as np
from OpenGL.GL import *

from world import Context, perspective
from models import Group
'''
from riglib.experiment import LogExperiment, traits
class Window(LogExperiment):
    background = (0,0,0)
    fps = 60

    def __init__(self, world, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.world = world
    
    def screen_init(self):
        pygame.init()

        flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.OPENGL
        pygame.display.set_mode((800,600), flags)
        
    def draw(self):
        pass
'''
class Test(threading.Thread):
    def __init__(self):
        super(Test, self).__init__()
        pygame.init()
        eye = [0, -2, 0.5]

        flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.OPENGL
        pygame.display.set_mode((800,600), flags)
        self.clock = pygame.time.Clock()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.5,0.6,0.5,1)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        self.ctx = Context(open("test.v.glsl"), open("test.f.glsl"))
        self.projection = perspective(45, 800./600, 0.0625, 256)
        #this effectively determines the modelview matrix
        self.world = Group([]).rotate_x(-90).translate(-eye[0], -eye[1], -eye[2])
        
    def add_model(self, model):
        self.world.add(model)
    
    def run(self):
        run = True
        glViewport(0,0,800,600)
        while run:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.ctx.program)

            self.ctx.uniforms['p_matrix'] = self.projection
            self.world.draw(self.ctx)

            pygame.display.flip()

            e = pygame.event.get(pygame.KEYDOWN)
            if len(e) > 0 and e[-1].key == 27:
                run = False
            self.clock.tick()

        pygame.display.quit()