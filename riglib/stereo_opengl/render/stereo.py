import numpy as np
from OpenGL.GL import *

from render import Renderer
from utils import offaxis_frusta

class LeftRight(Renderer):
    def __init__(self, window_size, fov, near, far, focal_dist, iod, **kwargs):
        w, h = window_size
        super(LeftRight, self).__init__((w/2,h), fov, near, far, **kwargs)
        self.projections = offaxis_frusta((w/2, h), fov, near, far, focal_dist, iod)
    
    def draw(self, root, **kwargs):
        w, h = self.size
        self.drawpos = 0,0
        glViewport(0, 0, w, h)
        super(LeftRight, self).draw(root, p_matrix=self.projections[0], **kwargs)
        self.drawpos = w,0
        glViewport(w, 0, w, h)
        super(LeftRight, self).draw(root, p_matrix=self.projections[1], **kwargs)

class Anaglyph(Renderer):
    def __init__(self, window_size, fov, near, far, focal_dist, iod, **kwargs):
        super(Anaglyph, self).__init__(window_size, fov, near, far, **kwargs)
        self.projections = offaxis_frusta(self.size, fov, near, far, focal_dist, iod)

    def draw(self, root, **kwargs):
        glViewport(0,0,self.size[0], self.size[1])
        glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_TRUE)
        super(Anaglyph, self).draw(root, p_matrix=self.projections[0], **kwargs)
        glClear(GL_DEPTH_BUFFER_BIT)
        glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE)
        super(Anaglyph, self).draw(root, p_matrix=self.projections[1], **kwargs)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

class MirrorDisplay(LeftRight):
    '''The mirror display requires a left-right flip, otherwise the sides are messed up'''
    def __init__(self, *args, **kwargs):
        super(MirrorDisplay, self).__init__(*args, **kwargs)
        xflip = np.eye(4)
        xflip[0,0] = -1

        self.projections = map(lambda x: np.dot(x, xflip), self.projections)