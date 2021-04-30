'''
Extensions of the render.Renderer class for stereo displays
'''

import numpy as np
from OpenGL.GL import *

from .render import Renderer
from ..utils import offaxis_frusta

class LeftRight(Renderer):
    def __init__(self, window_size, fov, near, far, focal_dist, iod, **kwargs):
        w, h = window_size
        super(LeftRight, self).__init__((w/2,h), fov, near, far, **kwargs)
        self.projections = offaxis_frusta((w/2, h), fov, near, far, focal_dist, iod)
    
    def draw(self, root, **kwargs):
        w, h = self.size
        glViewport(0, 0, w, h)
        super(LeftRight, self).draw(root, p_matrix=self.projections[0], **kwargs)
        glViewport(w, 0, w, h)
        super(LeftRight, self).draw(root, p_matrix=self.projections[1], **kwargs)

class RightLeft(LeftRight):
    def draw(self, root, **kwargs):
        w, h = self.size
        glViewport(w, 0, w, h)
        super(LeftRight, self).draw(root, p_matrix=self.projections[0], **kwargs)
        glViewport(0, 0, w, h)
        super(LeftRight, self).draw(root, p_matrix=self.projections[1], **kwargs)

class MirrorDisplay(Renderer):
    '''The mirror display requires a left-right flip, otherwise the sides are messed up'''
    def __init__(self, window_size, fov, near, far, focal_dist, iod, **kwargs):
        w, h = window_size
        super(MirrorDisplay, self).__init__((w/2,h), fov, near, far, **kwargs)
        flip = kwargs.pop('flip', True)
        self.projections = offaxis_frusta((w/2, h), fov, near, far, focal_dist, iod, flip=flip)
    
    def draw(self, root, **kwargs):
        '''
        Draw the 'root' model.

        Parameters
        ----------
        root: stereo_opengl.models.Model instance
            Root "world" model. Draws all submodels of this model.
        kwargs: optional keyword-arguments
            Optional shaders and stuff to pass to the lower-level drawing functions
        '''
        w, h = self.size
        w = int(w)
        h = int(h)

        # draw the portion of the screen with lower-left corner (0, 0), width 'w' and height 'h'
        glViewport(0, 0, w, h)
        super(MirrorDisplay, self).draw(root, p_matrix=self.projections[0], **kwargs)

        # draw the portion of the screen with lower-left corner (w, 0), width 'w' and height 'h'
        glViewport(w, 0, w, h)
        super(MirrorDisplay, self).draw(root, p_matrix=self.projections[1], **kwargs)

class DualMultisizeDisplay(Renderer):
    def __init__(self, main_window_size, mini_window_size, fov, near, far, focal_dist, iod, **kwargs):
        w, h = main_window_size
        w2, h2 = mini_window_size
        self.main_window_size = main_window_size
        self.mini_window_size = mini_window_size

        flip_main_z = kwargs.pop('flip_main_z', False)
        flip = kwargs.pop('flip', True)


        super(DualMultisizeDisplay, self).__init__((w+w2, h), fov, near, far, **kwargs)
        main_projections = offaxis_frusta(main_window_size, fov, near, far, focal_dist, iod, flip=flip, flip_z=flip_main_z)
        mini_projections = offaxis_frusta(mini_window_size, fov, near, far, focal_dist, iod, flip=flip)
        self.projections = (mini_projections[0], main_projections[0])

    def draw(self, root, **kwargs):
        '''
        Draw the 'root' model.

        Parameters
        ----------
        root: stereo_opengl.models.Model instance
            Root "world" model. Draws all submodels of this model.
        kwargs: optional keyword-arguments
            Optional shaders and stuff to pass to the lower-level drawing functions
        '''

        # draw the portion of the screen with lower-left corner (0, 0), width 'w' and height 'h'
        w2, h2 = self.mini_window_size
        glViewport(0, 0, w2, h2)
        super(DualMultisizeDisplay, self).draw(root, p_matrix=self.projections[0], **kwargs)

        # draw the portion of the screen with lower-left corner (w, 0), width 'w' and height 'h'
        w, h = self.main_window_size
        glViewport(w2, 0, w, h)
        super(DualMultisizeDisplay, self).draw(root, p_matrix=self.projections[1], **kwargs)

class Anaglyph(Renderer):
    '''
    From wikipedia: Anaglyph 3D is the name given to the stereoscopic 3D effect achieved by 
    means of encoding each eye's image using filters of different (usually chromatically 
    opposite) colors, typically red and cyan.
    '''
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
