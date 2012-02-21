from OpenGL.GL import *

def _make_shader(stype, source):
    shader = glCreateShader(stype)
    glShaderSource(shader, src)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        err = glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        raise Exception(err)
    return shader

class _getter(object):
    def __init__(self, type, prog):
        self.func = globals()['glGet{type}Location'.format(type=type)]
        self.prog = prog
    
    def __getattr__(self, attr):
        return self.func(self.prog, attr)
    
    def __getitem__(self, idx):
        return self.func(self.prog, idx)

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
    
        attributes = _getter("Attrib", self.program)
        uniforms = _getter("Uniform", self.program)
