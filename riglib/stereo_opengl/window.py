import pygame
from OpenGL.GL import *

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
    
    