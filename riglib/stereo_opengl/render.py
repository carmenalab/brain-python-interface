import os
import ctypes
import numpy as np
from OpenGL.GL import *

cwd = os.path.abspath(os.path.split(__file__)[0])
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
        for name, v in kwargs.items():
            if name in self.uniforms:
                self.uniforms[name] = v
            elif name in self.attributes:
                self.attributes[name] = v
            elif hasattr(v, "__call__"):
                v(self)

        for tex, funcs in models.items():
            if tex is not None:
                glUniform1i(self.uniforms.texture, ctx.texunits[tex][0])
            for drawfunc in funcs:
                drawfunc(self)

fbotypes = dict(
    depth=(GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT, GL_DEPTH_ATTACHMENT), 
    stencil=(GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, GL_STENCIL_ATTACHMENT), 
    color=(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, GL_COLOR_ATTACHMENT0)
)
class Renderer(object):
    def __init__(self, shaders, programs):
        self.shaders = dict()
        for k, v in shaders.items():
            print "Compiling shader %s..."%k
            self.add_shader(k, *v)
        
        self.programs = dict()
        for name, shaders in programs.items():
            self.add_program(name, shaders)
        
        maxtex = glGetIntegerv(GL_MAX_TEXTURE_COORDS)
        self.texavail = set((i, globals()['GL_TEXTURE%d'%i]) for i in range(maxtex))
        self.texunits = dict()

        self.fbos = dict()
        self.frametexs = dict()

        self.render_queue = None
    
    def _queue_render(self, root, shader=None):
        queue = dict((k, dict()) for k in self.programs.keys())

        for pname, drawfunc, tex in root.render_queue(shader=shader):
            if tex not in queue[pname]:
                queue[pname][tex] = []
            queue[pname][tex].append(drawfunc)
        
        for pname in self.programs.keys():
            assert len(self.texavail) > len(queue[pname])
            for tex in queue[pname].keys():
                if tex is not None:
                    self.add_texunit(tex)
        
        self.render_queue = queue
    
    def add_shader(self, name, stype, filename):
        src = open(os.path.join(cwd, "shaders", filename))
        shader = glCreateShader(stype)
        glShaderSource(shader, src)
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            err = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            raise Exception(err)
        
        self.shaders[name] = shader

    def add_texunit(self, tex):
        '''Input a Texture object, output a tuple (index, TexUnit)'''
        if tex not in self.texunits:
            unit = self.texavail.pop()
            glActiveTexture(unit[1])
            glBindTexture(GL_TEXTURE_2D, tex.tex)
            self.texunits[tex] = unit
        
        return self.texunits[tex]
    
    def add_program(self, name, shaders):
        shaders = [self.shaders[i] for i in shaders]
        sp = ShaderProgram(shaders)
        self.programs[name] = sp
    
    def draw(self, root, shader=None, **kwargs):
        if self.render_queue is None:
            self._queue_render(root, shader)
        
        for name, program in self.programs.items():
            program.draw(self, self.render_queue[name], **kwargs)
    
    def draw_to_fbo(self, kind, root, **kwargs):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbos[kind])
        #Erase old buffer info
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        self.draw(root, **kwargs)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def make_frametex(self, kind, frame_size):
        assert kind in fbotypes
        w, h = frame_size
        texform, textype, dtype, fbtype = fbotypes[kind]

        #First, create a texture
        frametex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, frametex)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexImage2D(GL_TEXTURE_2D, 0, texform, w, h, 0, textype, dtype, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.frametexs[kind] =  frametex
    
    def make_fbo(self, name, kinds):
        fbo = glGenFramebuffers(1)
        #Bind the texture to the renderer's framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, )
        for kind in kinds:
            texform, textype, dtype, fbtype = fbotypes[kind]
            glFramebufferTexture2D(GL_FRAMEBUFFER, fbtype, GL_TEXTURE_2D, self.frametexs[kind], 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.fbos[name] = fbo

def test():
    import pygame
    pygame.init()
    pygame.display.set_mode((100,100), pygame.OPENGL | pygame.DOUBLEBUF)

    return Renderer(
                shaders=dict(
                    passthru=(GL_VERTEX_SHADER, "passthrough.v.glsl"),
                    phong=(GL_FRAGMENT_SHADER, "phong_anaglyph.f.glsl")), 
                programs=dict(
                    default=("passthru", "phong"),
                )
            )