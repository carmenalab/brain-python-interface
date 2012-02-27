import functools
import Image

import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL import GLUT as glut

class Model(object):
    def __init__(self, shader="default", color=(0.5, 0.5, 0.5, 1), shininess=0.5):
        self.shader = shader
        self.xfm = np.eye(4)
        self.color = color
        self.shininess = shininess
    
    def init(self):
        pass
    
    def render_queue(self, xfm=np.eye(4), shader=None, **kwargs):
        '''Yields the shader, texture, and the partial drawfunc for queueing'''
        #Ignore other kwargs -- what do they mean?
        pfunc = functools.partial(self.draw, xfm=xfm)
        if shader is not None:
            yield shader, pfunc, None
        else:
            yield self.shader, pfunc, None
    
    def translate(self, x, y, z, reset=False):
        mat = np.array([[1,0,0,x],
                        [0,1,0,y],
                        [0,0,1,z],
                        [0,0,0,1]])
        if reset:
            self.xfm = mat
        else:
            self.xfm = np.dot(mat, self.xfm)
        return self
    
    def scale(self, x, y=None, z=None, reset=False):
        if y is None:
            y = x
        if z is None:
            z = x
        
        mat = np.array([[x,0,0,0],
                        [0,y,0,0],
                        [0,0,z,0],
                        [0,0,0,1]])
        
        if reset:
            self.xfm = mat
        else:
            self.xfm = np.dot(mat, self.xfm)
        return self
    
    def rotate_x(self, t, reset=False):
        t = np.radians(t)
        mat = np.array([[1,0,        0,         0],
                        [0,np.cos(t),-np.sin(t),0],
                        [0,np.sin(t), np.cos(t),0],
                        [0,0,        0,         1]])
        if reset:
            self.xfm = mat
        else:
            self.xfm = np.dot(mat, self.xfm)
        return self
    
    def rotate_y(self, t, reset=False):
        t = np.radians(t)
        mat = np.array([[ np.cos(t),0,np.sin(t),0],
                        [ 0,        1,0,        0],
                        [-np.sin(t),0,np.cos(t),0],
                        [ 0,0,      0,         1]])
        if reset:
            self.xfm = mat
        else:
            self.xfm = np.dot(mat, self.xfm)
        return self

    def rotate_z(self, t, reset=False):
        t = np.radians(t)
        mat = np.array([[np.cos(t),-np.sin(t),0,0],
                        [np.sin(t), np.cos(t),0,0],
                        [0,         0,        1,0],
                        [0,         0,        0,1]])
        if reset:
            self.xfm = mat
        else:
            self.xfm = np.dot(mat, self.xfm)

        return self

    def draw(self, ctx, xfm=np.eye(4), **kwargs):
        glUniformMatrix4fv(ctx.uniforms.xfm, 1, GL_TRUE, np.dot(xfm, self.xfm).astype(np.float32))
        glUniform4f(ctx.uniforms.basecolor, *self.color if "color" not in kwargs else kwargs['color'])
        glUniform1f(ctx.uniforms.shininess, self.shininess if "shininess" not in kwargs else kwargs['shininess'])

class Texture(object):
    def __init__(self, tex, 
        magfilter=GL_LINEAR, minfilter=GL_LINEAR, 
        wrap_x=GL_CLAMP_TO_EDGE, wrap_y=GL_CLAMP_TO_EDGE):
        self.opts = dict(magfilter=magfilter, minfilter=minfilter, wrap_x=wrap_x, wrap_y=wrap_y)
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
        self.texstr = tex
        self.size = size

    def init(self):
        gltex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, gltex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, self.opts['minfilter'])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, self.opts['magfilter'])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     self.opts['wrap_x'])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     self.opts['wrap_y'])
        glTexImage2D(
            GL_TEXTURE_2D, 0,           #target, level
            GL_RGB8,                    #internal format
            self.size[0], self.size[1], 0,      #width, height, border
            GL_RGBA, GL_UNSIGNED_BYTE,  #external format, type
            self.texstr                         #pixels
        )
        
        self.tex = gltex
    
    def set(self, idx):
        glActiveTexture(globals()['GL_ACTIVE%d'%idx])
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glUniform1i(ctx.uniforms['texture'], idx)

class TexModel(Model):
    def __init__(self, tex=None, **kwargs):
        if tex is not None:
            kwargs['color'] = (0,0,0,1)
        super(TexModel, self).__init__(**kwargs)
        if isinstance(tex, Texture):
            #only single texture, assume it's full weight
            tex = [(tex, (1.,1.,1.,1.))]
        
        self.texs = tex
    
    def render_queue(self, xfm=np.eye(4), shader=None, **kwargs):
        #Ignore other kwargs -- what do they mean?
        l = len(self.texs) if self.texs is not None else 0
        pfunc = functools.partial(self.draw, xfm=xfm)
        if shader is not None:
            yield shader, pfunc, l
        else:
            yield self.shader, pfunc, l

class Group(Model):
    def __init__(self, models):
        super(Group, self).__init__()
        self.models = models

    def init(self):
        for model in self.models:
            model.init()
    
    def render_queue(self, xfm=np.eye(4), **kwargs):
        for model in self.models:
            for out in model.render_queue(np.dot(xfm, self.xfm), **kwargs):
                yield out
    
    def draw(self, ctx, xfm=np.eye(4), **kwargs):
        for model in self.models:
            model.draw(ctx, np.dot(xfm, self.xfm), **kwargs)
    
    def add(self, model):
        self.models.append(model)
    
    def __getitem__(self, idx):
        return self.models[idx]
    

builtins = dict([ (n[9:].lower(), getattr(glut, n)) 
                    for n in dir(glut) 
                    if "glutSolid" in n])
class Builtins(Model):
    def __init__(self, model, xfm=np.eye(4), *args):
        super(Builtins, self).__init__(xfm)
        assert model in builtins
        self.model = builtins['model']
        self.args = args
    
    def draw(self, ctx, xfm=np.eye(4)):
        glPushMatrix()
        glLoadMatrixf(np.dot(xfm, self.xfm).ravel())
        self.model(*self.args)
        glPopMatrix()

class TriMesh(TexModel):
    '''Basic triangle mesh model. Houses the GL functions for making buffers and displaying triangles'''
    def __init__(self, verts, polys, normals=None, tcoords=None, **kwargs):
        super(TriMesh, self).__init__(**kwargs)
        if verts.shape[1] == 3:
            verts = np.hstack([verts, np.ones((len(verts),1))])
        if normals.shape[1] == 3:
            normals = np.hstack([normals, np.ones((len(normals),1))])

        self.verts = verts
        self.polys = polys
        self.tcoords = tcoords
        self.normals = normals
    
    def init(self):
        self.vbuf = glGenBuffers(1)
        self.ebuf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbuf)
        glBufferData(GL_ARRAY_BUFFER, 
            self.verts.astype(np.float32).ravel(), GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebuf)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
            self.polys.astype(np.uint16).ravel(), GL_STATIC_DRAW)

        if self.tcoords is not None:
            self.tbuf = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.tbuf)
            glBufferData(GL_ARRAY_BUFFER, 
                self.tcoords.astype(np.float32).ravel(), GL_STATIC_DRAW)
        
        if self.normals is not None:
            self.nbuf = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.nbuf)
            glBufferData(GL_ARRAY_BUFFER,
                self.normals.astype(np.float32).ravel(), GL_STATIC_DRAW)
    
    def draw(self, ctx, xfm=np.eye(4)):
        super(TriMesh, self).draw(ctx, xfm)
        glEnableVertexAttribArray(ctx.attributes['position'])
        glBindBuffer(GL_ARRAY_BUFFER, self.vbuf)
        glVertexAttribPointer( ctx.attributes['position'],
            4, GL_FLOAT, GL_FALSE, 4*4, GLvoidp(0))
        
        if self.tcoords is not None and ctx.attributes['texcoord'] != -1:
            glEnableVertexAttribArray(ctx.attributes['texcoord'])
            glBindBuffer(GL_ARRAY_BUFFER, self.tbuf)
            glVertexAttribPointer(
                ctx.attributes['texcoord'], 2, 
                GL_FLOAT, GL_FALSE, 4*2, GLvoidp(0))

        if self.normals is not None and ctx.attributes['normal'] != -1:
            glEnableVertexAttribArray(ctx.attributes['normal'])
            glBindBuffer(GL_ARRAY_BUFFER, self.nbuf)
            glVertexAttribPointer(
                ctx.attributes['normal'], 4, 
                GL_FLOAT, GL_FALSE, 4*4, GLvoidp(0))
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebuf);
        glDrawElements(
            GL_TRIANGLES,           # mode
            len(self.polys)*3,      # count
            GL_UNSIGNED_SHORT,      # type
            GLvoidp(0)              # element array buffer offset
        )
        glDisableVertexAttribArray(ctx.attributes['position'])
        if self.tcoords is not None and ctx.attributes['texcoord'] != -1:
            glDisableVertexAttribArray(ctx.attributes['texcoord'])
        if self.normals is not None and ctx.attributes['normal'] != -1:
            glDisableVertexAttribArray(ctx.attributes['normal'])

class FlatMesh(TriMesh):
    '''Takes smoothed or no-normal meshes and gives them a flat shading'''
    def __init__(self, verts, polys, normals=None, **kwargs):
        checked = dict()
        normals = []
        nverts = []
        npolys = []

        for i, poly in enumerate(polys):
            v1 = verts[poly[1]] - verts[poly[0]]
            v2 = verts[poly[2]] - verts[poly[0]]
            nvec = tuple(np.cross(v1, v2))

            npoly = []
            for v in poly:
                vert = tuple(verts[v])
                if (vert, nvec) not in checked:
                    checked[(vert, nvec)] = len(nverts)
                    npoly.append(len(nverts))
                    nverts.append(vert)
                    normals.append(nvec)
                else:
                    npoly.append(checked[(vert, nvec)])

            npolys.append(npoly)
        

        super(FlatMesh, self).__init__(np.array(nverts), np.array(npolys), 
            normals=np.array(normals), **kwargs)
