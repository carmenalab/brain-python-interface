'''Needs docs'''


import numpy as np
from OpenGL.GL import *

from .models import Model

textypes = {GL_UNSIGNED_BYTE:np.uint8, GL_FLOAT:np.float32}
class Texture(object):
    def __init__(self, tex, size=None,
        magfilter=GL_LINEAR, minfilter=GL_LINEAR, 
        wrap_x=GL_CLAMP_TO_EDGE, wrap_y=GL_CLAMP_TO_EDGE,
        iformat=GL_RGBA8, exformat=GL_RGBA, dtype=GL_UNSIGNED_BYTE):

        self.opts = dict(
            magfilter=magfilter, minfilter=minfilter, 
            wrap_x=wrap_x, wrap_y=wrap_y,
            iformat=iformat, exformat=exformat, dtype=dtype)

        if isinstance(tex, np.ndarray):
            if tex.max() <= 1:
                tex *= 255
            if len(tex.shape) < 3:
                tex = np.tile(tex, [3, 1, 1]).T
            if tex.shape[-1] == 3:
                tex = np.dstack([tex, np.ones(tex.shape[:-1])])
            size = tex.shape[:2]
            tex = tex.astype(np.uint8).tostring()
        elif isinstance(tex, str):
            im = pygame.image.load(tex)
            size = tex.get_size()
            tex = pygame.image.tostring(im, 'RGBA')
        
        self.texstr = tex
        self.size = size
        self.tex = None

    def init(self):
        gltex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, gltex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, self.opts['minfilter'])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, self.opts['magfilter'])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     self.opts['wrap_x'])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     self.opts['wrap_y'])
        glTexImage2D(
            GL_TEXTURE_2D, 0,                           #target, level
            self.opts['iformat'],                       #internal format
            self.size[0], self.size[1], 0,              #width, height, border
            self.opts['exformat'], self.opts['dtype'],  #external format, type
            self.texstr if self.texstr is not None else 0   #pixels
        )
        
        self.tex = gltex
    
    def set(self, idx):
        glActiveTexture(GL_TEXTURE0+idx)
        glBindTexture(GL_TEXTURE_2D, self.tex)
    
    def get(self, filename=None):
        current = glGetInteger(GL_TEXTURE_BINDING_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        texstr = glGetTexImage(GL_TEXTURE_2D, 0, self.opts['exformat'], self.opts['dtype'])
        glBindTexture(GL_TEXTURE_2D, current)
        im = np.fromstring(texstr, dtype=textypes[self.opts['dtype']])
        im.shape = (self.size[1], self.size[0], -1)
        if filename is not None:
            np.save(filename, im)
        return im


class MultiTex(object):
    '''This is not ready yet!'''
    def __init__(self, textures, weights):
        raise NotImplementedError
        assert len(textures) < max_multitex
        self.texs = textures
        self.weights = weights

class TexModel(Model):
    def __init__(self, tex=None, **kwargs):
        if tex is not None:
            kwargs['color'] = (0,0,0,1)
        super(TexModel, self).__init__(**kwargs)
        
        self.tex = tex
    
    def init(self):
        super(TexModel, self).init()
        if self.tex.tex is None:
            self.tex.init()
        
    def render_queue(self, shader=None, **kwargs):
        if shader is not None:
            yield shader, self.draw, self.tex
        else:
            yield self.shader, self.draw, self.tex