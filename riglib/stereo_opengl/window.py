import threading

import pygame
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
        pygame.init()
        eye = [0, 0, -5]

        flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.OPENGL
        pygame.display.set_mode((800,600), flags)
        self.clock = pygame.time.Clock()

        glEnable(GL_BLEND)
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0,0,0,1)

        self.ctx = Context(open("test.v.glsl"), open("test.f.glsl"))
        self.ctx.uniforms['p_matrix'] = perspective(30, 600./800, 0.25, 256)
        #this effectively determines the modelview matrix
        self.world = Group(self.ctx, []).translate(-eye[0], -eye[1], -eye[2]) 
    
    def add_model(self, model):
        self.world.add(model)
    
    def run(self):
        run = True
        while run:
            glClear()
            for side in ['l', 'r']:
                glViewport((0,400)[side=="r"],0,400,600)
                glUseProgram(self.ctx.program)
                self.world.draw()
            pygame.display.flip()
            e = pygame.event.get(pygame.KEYDOWN)
            if len(e) > 0 and e[-1].key == 27:
                run = False
            self.clock.tick()