import Image
import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL import GLUT as glut
from scipy.spatial import Delaunay

class Model(object):
    def __init__(self, ctx, xfm=np.eye(4)):
        self.xfm = xfm
        self.ctx = ctx
    
    def translate(self, x, y, z):
        mat = np.array([[1,0,0,x],
                        [0,1,0,y],
                        [0,0,1,z],
                        [0,0,0,1]])
        self.xfm = np.dot(self.xfm, mat)
        return self
    
    def scale(self, x, y=None, z=None):
        if y is None:
            y = x
        if z is None:
            z = x

        mat = np.array([[x,0,0,0],
                        [0,y,0,0],
                        [0,0,z,0],
                        [0,0,0,1]])
        self.xfm = np.dot(self.xfm, mat)
        return self
    
    def rotate_x(self, t):
        mat = np.array([[1,0,        0,         0],
                        [0,np.cos(t),-np.sin(t),0],
                        [0,np.sin(t), np.cos(t),0],
                        [0,0,        0,         1]])
        self.xfm = np.dot(self.xfm, mat)
        return self
    
    def rotate_y(self, t):
        mat = np.array([[ np.cos(t),0,np.sin(t),0],
                        [ 0,        1,0,        0],
                        [-np.sin(t),0,np.cos(t),0],
                        [ 0,0,      0,         1]])
        self.xfm = np.dot(self.xfm, mat)
        return self

    def rotate_z(self, t):
        mat = np.array([[np.cos(t),-np.sin(t),0,0],
                        [np.sin(t), np.cos(t),0,0],
                        [0,         0,        1,0],
                        [0,         0,        0,1]])
        self.xfm = np.dot(self.xfm, mat)
        return self

    def draw(self, xfm=np.eye(4)):
        self.ctx.uniforms.xfm = self.xfm

class Texture2D(object):
    def __init__(self, ctx, tex, 
        magfilter=GL_LINEAR, 
        minfilter=GL_LINEAR, 
        wrap_x=GL_CLAMP_TO_EDGE, 
        wrap_y=GL_CLAMP_TO_EDGE):

        if isinstance(tex, np.ndarray):
            if tex.max() <= 1:
                tex *= 255
            if len(tex.shape) < 3:
                tex = np.tile(tex, [1,1,4])
            size = tex.shape[:2]
            tex = tex.astype(np.uint8).tostring()
        elif isinstance(tex, Image):
            size = tex.size
            tex = tex.tostring()
        elif isinstance(tex, str):
            im = pygame.image.load(tex)
            size = tex.get_size()
            tex = pygame.image.tostring(im, 'RGBA')
        
        gltex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, gltex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minfilter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magfilter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     wrap_x);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     wrap_y);
        glTexImage2D(
            GL_TEXTURE_2D, 0,           #target, level
            GL_RGB8,                    #internal format
            size[0], size[1], 0,      #width, height, border
            GL_RGBA, GL_UNSIGNED_BYTE,  #external format, type
            tex                         #pixels
        )
        
        self.ctx = ctx
        self.tex = gltex
        self.size = size

    def set(self):
        glActiveTexture(GL_TEXTURE0)
        self.ctx.uniforms['texture'] = 0
        glBindTexture(GL_TEXTURE_2D, self.tex)

class TexModel(Model):
    def __init__(self, ctx, tex, xfm=np.eye(4)):
        super(TexModel, self).__init__(ctx, xfm)
        self.tex = tex
    
    def draw(self, xfm=np.eye(4)):
        self.tex.set()
        super(TexModel, self).draw(xfm)


class Group(Model):
    def __init__(self, ctx, models, xfm=np.eye(4)):
        super(Group, self).__init__(ctx, xfm)
        self.models = models
    
    def draw(self, xfm=np.eye(4)):
        for model in self.models:
            model.draw(np.dot(xfm, self.xfm))
    
    def add(self, model):
        self.model.append(model)
    

builtins = dict([ (n[9:].lower(), getattr(glut, n)) 
                    for n in dir(glut) 
                    if "glutSolid" in n])
class Builtins(Model):
    def __init__(self, ctx, model, xfm=np.eye(4), *args):
        super(Builtins, self).__init__(ctx, xfm)
        assert model in builtins
        self.model = builtins['model']
        self.args = args
    
    def draw(self, xfm=np.eye(4)):
        glPushMatrix()
        glLoadMatrixf(np.dot(xfm, self.xfm).ravel())
        self.model(*self.args)
        glPopMatrix()

class TriMesh(Model):
    def __init__(self, ctx, verts, polys, tcoords=None, xfm=np.eye(4)):
        super(TriMesh, self).__init__(ctx, xfm)
        if verts.shape[1] == 3:
            verts = np.hstack([verts, np.ones(len(verts),)])

        self.verts = verts
        self.polys = polys
        self.tcoords = tcoords

        self.vbuf = glGenBuffers(1)
        self.ebuf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbuf)
        glBufferData(GL_ARRAY_BUFFER, self.verts.ravel(), GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebuf)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.polys.ravel(), GL_STATIC_DRAW)

        if tcoords is not None:
            self.tbuf = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.tbuf)
            glBufferData(GL_ARRAY_BUFFER, self.tcoords.ravel(), GL_STATIC_DRAW)
    
    def draw(self, xfm=np.eye(4)):
        super(TriMesh, self).draw(xfm)
        glEnableVertexAttribArray(self.ctx.attributes['vert'])
        glBindBuffer(GL_ARRAY_BUFFER, self.vbuf)
        glVertexAttribPointer(
            self.ctx.attributes['vert'],# attribute
            len(self.verts),            # size
            GL_FLOAT,                   # type
            GL_FALSE,                   # normalized?
            32*4,                       # stride
            0                           # array buffer offset
        )

        if self.tcoords is not None:
            glEnableVertexAttribArray(self.ctx.attributes['texcoords'])
            glBindBuffer(GL_ARRAY_BUFFER, self.tbuf)
            glVertexAttribPointer(
                self.ctx.attributes['texcoords'], len(self.tcoords), 
                GL_FLOAT, GL_FALSE, 32*2, 0
            )

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebuf);
        glDrawElements(
            GL_TRIANGLES,           # mode
            len(self.polys),        # count
            GL_UNSIGNED_SHORT,      # type
            0                       # element array buffer offset
        )
        glDisableVertexAttribArray(self.ctx.attributes['vert'])
        if self.tcoords is not None:
            glDisableVertexAttribArray(self.ctx.attributes['texcoords'])

class PolyMesh(TriMesh):
    '''This model accepts arbitrary polygons. It first triangulates all polys
    then submits it to TriMesh'''
    def __init__(self, ctx, verts, polys, tcoords=None, xfm=np.eye(4)):
        plist = []
        for poly in polys:
            if len(poly) > 3:
                #We have a non-triangle, let's triangulate it
                dl = Delaunay(verts[tuple(poly), :])
                for p in dl.vertices:
                    plist.append([poly[i] for i in p])
            else:
                plist.append(poly)
        
        super(PolyMesh, self).__init__(ctx, verts, np.array(plist), tcoords, xfm)