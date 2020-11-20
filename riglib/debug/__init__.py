import pygame
import numpy as np
from riglib.source import DataSourceSystem

class LFP(DataSourceSystem):
    '''
    2D mouse position emulating a neural data source
    '''
    dtype = np.dtype('float')
    update_freq = 100

    def start(self):
        self.pos = np.zeros(2)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def stop(self):
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

    def get(self):
        rel = pygame.mouse.get_rel()
        self.pos += rel
        return self.pos