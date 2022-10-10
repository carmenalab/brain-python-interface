'''
Collections of shapes to render on screen
'''

import numpy as np
from OpenGL.GL import *
from OpenGL import GLUT as glut

from .xfm import Transform

class Model(object):
    def __init__(self, shader="default", color=(0.5, 0.5, 0.5, 1), 
        shininess=10, specular_color=(1.,1.,1.,1.)):
        '''
        Docstring Constructor for Model

        Parameters
        ----------
        shader: string
            OpenGL shading method?
        color: tuple of length 4
            (r, g, b, translucence)
        shininess: float
            ???
        specular_color: tuple of length 4
            color of the light source???
        '''
        self.shader = shader
        self.parent = None

        # This is different from self.xfm = Transform() for children of Model implemented with multiple inheritance
        super(Model, self).__setattr__("xfm", Transform())
        self.color = color
        self.shininess = shininess
        self.spec_color = specular_color

        # The orientation of the object, in the world frame
        self._xfm = self.xfm
        self.allocated = False
    
    def __setattr__(self, attr, xfm):
        '''Checks if the xfm was changed, and recaches the _xfm which is sent to the shader'''
        val = super(Model, self).__setattr__(attr, xfm)
        if attr == "xfm":
            self._recache_xfm()

        return val
    
    def _recache_xfm(self):
        '''
        For models with a parent, the transform of the current model must be cascaded with the parent model's transform.
        NOTE: this only goes one level up the graphics tree, so the transform is 
        always with respect to the parent's frame, not with respect to the world frame!
        '''
        if self.parent is not None:
            self._xfm = self.parent._xfm * self.xfm
        else:
            self._xfm = self.xfm
    
    def init(self):
        allocated = self.allocated
        self.allocated = True
        return allocated

    def rotate_x(self, deg, reset=False):
        self.xfm.rotate_x(np.radians(deg), reset=reset)
        self._recache_xfm()
        return self

    def rotate_y(self, deg, reset=False):
        self.xfm.rotate_y(np.radians(deg), reset=reset)
        self._recache_xfm()
        return self

    def rotate_z(self, deg, reset=False):
        self.xfm.rotate_z(np.radians(deg), reset=reset)
        self._recache_xfm()
        return self

    def translate(self, x, y, z, reset=False):
        self.xfm.translate(x,y,z, reset=reset)
        self._recache_xfm()
        return self
    
    def render_queue(self, shader=None):
        '''Yields the shader, texture, and the partial drawfunc for queueing'''
        if shader is not None:
            yield shader, self.draw, None
        else:
            yield self.shader, self.draw, None

    def draw(self, ctx, **kwargs):
        '''
        Parameters
        ----------
        ctx: ??????
        kwargs: optional keyword arguments
            Can specify 'color', 'specular_color', or 'shininess' of the object to draw (overriding the model's attributes)

        Returns: None
        '''
        
        glUniformMatrix4fv(ctx.uniforms.xfm, 1, GL_TRUE, self._xfm.to_mat().astype(np.float32))
        glUniform4f(ctx.uniforms.basecolor, *kwargs.pop('color', self.color))
        glUniform4f(ctx.uniforms.spec_color, *kwargs.pop('specular_color', self.spec_color))
        glUniform1f(ctx.uniforms.shininess, kwargs.pop('shininess', self.shininess))

        # glUniform4f(ctx.uniforms.basecolor, *(self.color if "color" not in kwargs else kwargs['color']))
        # glUniform4f(ctx.uniforms.spec_color, *(self.spec_color if "specular_color" not in kwargs else kwargs['spec_color']))
        # glUniform1f(ctx.uniforms.shininess, self.shininess if "shininess" not in kwargs else kwargs['shininess'])        

    def attach(self):
        assert self.parent is not None
        while self not in self.parent.models:
            self.parent.models.append(self)

    def detach(self):
        assert self.parent is not None
        while self in self.parent.models:
            self.parent.models.remove(self)


class Group(Model):
    def __init__(self, models=()):
        super(Group, self).__init__()
        self.models = []
        for model in models:
            self.add(model)
    
    def add(self, model):
        self.models.append(model)
        model.parent = self
        model._recache_xfm()

    def remove(self, model):
        # remove the redundant models
        if model not in self.models:
            return
        self.models.remove(model)
        del model

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
        allocated = super(TriMesh, self).init()
        if not allocated:
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
        return allocated
    
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

class PolyMesh(TriMesh):
    def __init__(self, verts, polys, **kwargs):
        tripoly = []
        for poly in polys:
            for p in zip(poly[1:-1], poly[2:]):
                tripoly.append((poly[1],)+p)
        super(PolyMesh, self).__init__(verts, tripoly, **kwargs)


def obj_load(filename):

    facesplit = lambda x: x.split('/')
    verts, polys, normals, tcoords = [], [], [], []
    objfile = open(filename)
    for line in objfile:
        el = line.split()
        if el[0] == "#":
            pass
        elif el[0] == "v":
            verts.append(list(map(float, el[1:])))
        elif el[0] == "vt":
            tcoords.append(list(map(float, el[1:])))
        elif el[0] == "vn":
            normals.append(list(map(float, el[1:])))
        elif el[0] == "f":
            for v in el[1:]:
                pass
            list(map(facesplit, el[1:]))
