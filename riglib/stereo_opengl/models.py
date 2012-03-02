import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL import GLUT as glut

class Model(object):
    def __init__(self, shader="default", color=(0.5, 0.5, 0.5, 1), 
        shininess=0.5, specular_color=(1.,1.,1.,1.)):
        self.shader = shader
        self.parent = None
        self.xfm = np.eye(4)
        self.color = color
        self.shininess = shininess
        self.spec_color = specular_color

        self._xfm = self.xfm
    
    def __setattr__(self, attr, xfm):
        '''Checks if the xfm was changed, and updates its xfm'''
        val = super(Model, self).__setattr__(attr, xfm)
        if attr == "xfm":
            self._recache_xfm()

        return val
    
    def _recache_xfm(self):
        if self.parent is not None:
            self._xfm = np.dot(self.parent._xfm, self.xfm)
        else:
            self._xfm = self.xfm
    
    def init(self):
        pass
    
    def render_queue(self, shader=None):
        '''Yields the shader, texture, and the partial drawfunc for queueing'''
        if shader is not None:
            yield shader, self.draw, None
        else:
            yield self.shader, self.draw, None
    
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

    def draw(self, ctx, **kwargs):
        glUniformMatrix4fv(ctx.uniforms.xfm, 1, GL_TRUE, self._xfm.astype(np.float32))
        glUniform4f(ctx.uniforms.basecolor, *(self.color if "color" not in kwargs else kwargs['color']))
        glUniform4f(ctx.uniforms.spec_color, *(self.spec_color if "specular_color" not in kwargs else kwargs['spec_color']))
        glUniform1f(ctx.uniforms.shininess, self.shininess if "shininess" not in kwargs else kwargs['shininess'])

class Group(Model):
    def __init__(self, models=()):
        self._xfm = np.eye(4)
        self.models = []
        for model in models:
            self.add(model)
        super(Group, self).__init__()

    def init(self):
        for model in self.models:
            model.init()
    
    def render_queue(self, xfm=np.eye(4), **kwargs):
        for model in self.models:
            for out in model.render_queue(**kwargs):
                yield out
    
    def draw(self, ctx, **kwargs):
        for model in self.models:
            model.draw(ctx, **kwargs)
    
    def add(self, model):
        self.models.append(model)
        model.parent = self
        model._recache_xfm()
    
    def __getitem__(self, idx):
        return self.models[idx]
    
    def _recache_xfm(self):
        super(Group, self)._recache_xfm()
        for model in self.models:
            model._recache_xfm()
    

builtins = dict([ (n[9:].lower(), getattr(glut, n)) 
                    for n in dir(glut) 
                    if "glutSolid" in n])
class Builtins(Model):
    def __init__(self, model, shader="fixedfunc", *args):
        super(Builtins, self).__init__(xfm)
        assert model in builtins
        self.model = builtins['model']
        self.args = args
    
    def draw(self, ctx):
        glPushMatrix()
        glLoadMatrixf(np.dot(xfm, self.xfm).ravel())
        self.model(*self.args)
        glPopMatrix()

class TriMesh(Model):   
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
        super(TriMesh, self).init()
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
    
    def draw(self, ctx):
        super(TriMesh, self).draw(ctx)
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
