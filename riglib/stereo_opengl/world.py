import numpy as np
from OpenGL.GL import *

def frustum(l, r, u, b, n, f):
    '''Emulates glFrustum'''
    rl, nrl = r + l, r - l
    tb, ntb = t + b, t - b
    fn, nfn = f + n, f - n
    return np.array([[2*n / nrl, 0, rl / nrl, 0],
                     [0, 2*n / ntb, tb / ntb, 0],
                     [0,0,-fn / nfn, -2*f*n / nfn],
                     [0,0,-1,0]])

def perspective(angle, aspect, near, far):
    '''Generates a perspective transform matrix'''
    ta = np.tan(np.radians(angle))
    fn, nfn = far + near, far - near
    return np.array([[1./ta, 0,0,0],
                     [0, aspect/ta, 0,0],
                     [0,0,fn/nfn, -2*far*near/nfn],
                     [0,0,1,0]])

def _make_shader(stype, src):
    shader = glCreateShader(stype)
    glShaderSource(shader, src)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        err = glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        raise Exception(err)
    return shader

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
        super(_getter, self).__setattr__("func", globals()['glGet{type}Location'.format(type=type)])
        super(_getter, self).__setattr__("prog", prog)
        super(_getter, self).__setattr__("cache", dict())
    
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
                func = globals()['glUniform%d%sv'%(len(val[0]), t)]
                func(self.cache[attr], len(val), val)
        elif isinstance(val, (int, float)): 
            #single value, push with glUniform1
            globals()['glUniform1%s'%_typename[type(val)]](self.cache[attr], val)

class Context(object):
    def __init__(self, vshade, fshade):
        self.vshade = _make_shader(GL_VERTEX_SHADER, vshade.read())
        self.fshade = _make_shader(GL_FRAGMENT_SHADER, fshade.read())

        self.program = glCreateProgram()
        glAttachShader(self.program, self.vshade)
        glAttachShader(self.program, self.fshade)
        glLinkProgram(self.program)

        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            err = glGetProgramInfoLog(self.program)
            glDeleteProgram(self.program)
            raise Exception(err)
    
        self.attributes = _getter("Attrib", self.program)
        self.uniforms = _getter("Uniform", self.program)
