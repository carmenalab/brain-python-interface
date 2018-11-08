'''Needs docs'''


import numpy as np
from OpenGL.GL import *

from ..textures import Texture

_mattypes = {
    (4,4):"4", (3,3):"3", (2,2):"2", 
    (2,3):"2x3",(3,2):"3x2",
    (2,4):"2x4",(4,2):"4x2",
    (3,4):"3x4",(4,3):"4x3"
}
_typename = { int:"i", float:"f" }
class _getter(object):
    '''Wrapper object which allows direct getting and setting of shader values'''
    def __init__(self, type, prog):
        setter = super(_getter, self).__setattr__
        
        setter("prog", prog)
        setter("cache", dict())
        setter("func", globals()['glGet{type}Location'.format(type=type)])

        if type == "Uniform":
            setter("info", dict())
            for i in range(glGetProgramiv(self.prog, GL_ACTIVE_UNIFORMS)):
                name, size, t = glGetActiveUniform(self.prog, i)
                self.info[name] = t
    
    def __getattr__(self, attr):
        if attr not in self.cache:
            self.cache[attr] = self.func(self.prog, attr)
        return self.cache[attr]
    
    def __getitem__(self, attr):
        if attr not in self.cache:
            self.cache[attr] = self.func(self.prog, attr)
        return self.cache[attr]

    def __setitem__(self, attr, val):
        self._set(attr, val)
    
    def __setattr__(self, attr, val):
        self._set(attr, val)
    
    def __contains__(self, attr):
        if attr not in self.cache:
            self.cache[attr] = self.func(self.prog, attr)
        return self.cache[attr] != -1
    
    def _set(self, attr, val):
        '''This heinously complicated function has to guess the function to use because
        there are no strong types in python, hence we just have to guess'''
        if attr not in self.cache:
            self.cache[attr] = self.func(self.prog, attr)

        if isinstance(val, np.ndarray) and len(val.shape) > 1:
            assert len(val.shape) <= 3
            if val.shape[-2:] in _mattypes:
                nmats = val.shape[0] if len(val.shape) == 3 else 1
                fname = _mattypes[val.shape[-2:]]
                func = globals()['glUniformMatrix%sfv'%fname]
                #We need to transpose all numpy matrices since numpy is row-major
                #and opengl is column-major
                func(self.cache[attr], nmats, GL_TRUE, val.astype(np.float32).ravel())
        elif isinstance(val, (list, tuple, np.ndarray)):
            #glUniform\d[if]v
            if isinstance(val[0], (tuple, list, np.ndarray)):
                assert len(val[0]) <= 4
                t = _typename[type(val[0][0])]
                func = globals()['glUniform%d%sv'%(len(val[0]), t)]
                func(self.cache[attr], len(val), np.array(val).astype(np.float32).ravel())
            else:
                t = _typename[type(val[0])]
                func = globals()['glUniform%d%s'%(len(val), t)]
                func(self.cache[attr], *val)
        elif isinstance(val, (int, float)): 
            #single value, push with glUni2form1
            globals()['glUniform1%s'%_typename[type(val)]](self.cache[attr], val)

class ShaderProgram(object):
    def __init__(self, shaders):
        self.shaders = shaders
        self.program = glCreateProgram()
        for shader in shaders:
            glAttachShader(self.program, shader)
        glLinkProgram(self.program)

        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            err = glGetProgramInfoLog(self.program)
            glDeleteProgram(self.program)
            raise Exception(err)
        
        self.attributes = _getter("Attrib", self.program)
        self.uniforms = _getter("Uniform", self.program)
    
    def draw(self, ctx, models, **kwargs):
        glUseProgram(self.program)
        for name, v in list(kwargs.items()):
            if isinstance(v, Texture):
                self.uniforms[name] = ctx.get_texunit(v)
            elif name in self.uniforms:
                self.uniforms[name] = v
            elif name in self.attributes:
                self.attributes[name] = v
            elif hasattr(v, "__call__"):
                v(self)
        
        for tex, funcs in list(models.items()):
            if tex is None:
                self.uniforms.texture = ctx.get_texunit("None")
            else:
                self.uniforms.texture = ctx.get_texunit(tex)

            for drawfunc in funcs:
                drawfunc(self)
