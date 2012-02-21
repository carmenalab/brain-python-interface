from OpenGL.GL import *

class World(object):
    def __init__(self, models):
        self.models = models
    
    def draw(self):
        for model in self.models:
            model.draw()