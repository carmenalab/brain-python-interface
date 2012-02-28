import os
import ctypes
import numpy as np
from OpenGL.GL import *

from models import Texture

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
            else:
                glUniform1i(self.uniforms.texture, 0)
            for drawfunc in funcs:
                drawfunc(self)

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
        #Use first texture unit as the "blank" texture
        self.texavail = set((i, globals()['GL_TEXTURE%d'%i]) for i in range(1, maxtex))
        self.texunits = dict()

        self.fbos = dict()
        self.frametexs = dict()

        self.render_queue = None

        vbuf = glGenBuffers(1)
        ebuf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbuf)
        glBufferData(GL_ARRAY_BUFFER, np.array([(-1,-1), (1,-1), (1,1), (-1,1)]).astype(np.float32), GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebuf)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array([(0,1,3),(1,2,3)]).astype(np.uint16).ravel(), GL_STATIC_DRAW)
        self.fsquad_buf = vbuf, ebuf
    
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
                    self.get_texunit(tex)
        
        self.render_queue = queue
    
    def get_texunit(self, tex):
        '''Input a Texture object, output a tuple (index, TexUnit)'''
        if tex not in self.texunits:
            unit = self.texavail.pop()
            glActiveTexture(unit[1])
            glBindTexture(GL_TEXTURE_2D, tex.tex)
            self.texunits[tex] = unit
        
        return self.texunits[tex]
    
    def add_shader(self, name, stype, filename, *includes):
        src = []
        main = open(os.path.join(cwd, "shaders", filename))
        version = main.readline().strip()
        for inc in includes:
            incfile = open(os.path.join(cwd, "shaders", inc))
            ver = incfile.readline().strip()
            assert ver == version, "Version: %s, %s"%(ver, version)
            src.append(incfile.read())
            incfile.close()
        src.append(main.read())
        main.close()

        shader = glCreateShader(stype)
        glShaderSource(shader, "\n".join(src))
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            err = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            raise Exception(err)
        
        self.shaders[name] = shader
    
    def add_program(self, name, shaders):
        shaders = [self.shaders[i] for i in shaders]
        sp = ShaderProgram(shaders)
        self.programs[name] = sp
    
    def draw(self, root, shader=None, **kwargs):
        if self.render_queue is None:
            self._queue_render(root, shader)
        
        for name, program in self.programs.items():
            program.draw(self, self.render_queue[name], **kwargs)
    
    def draw_to_fbo(self, fbo, root, **kwargs):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo)
        glDrawBuffers(fbo.types)
        #Erase old buffer info
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        self.draw(root, **kwargs)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def draw_fsquad(self, shader, **kwargs):
        ctx = self.programs[shader]
        glUseProgram(ctx.program)
        for name, arg in kwargs.items():
            ctx.uniforms[name] = arg
        
        glEnableVertexAttribArray(ctx.attributes['position'])
        glBindBuffer(GL_ARRAY_BUFFER, self.fsquad_buf[0])
        glVertexAttribPointer(ctx.attributes['position'],
            4, GL_FLOAT, GL_FALSE, 4*4, GLvoidp(0))
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.fsquad_buf[1]);
        glDrawElements(
            GL_TRIANGLES,           # mode
            6,      # count
            GL_UNSIGNED_SHORT,      # type
            GLvoidp(0)              # element array buffer offset
        )
        glDisableVertexAttribArray(ctx.attributes['position'])


fbotypes = dict(
    depth=(GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT, GL_DEPTH_ATTACHMENT), 
    stencil=(GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, GL_STENCIL_ATTACHMENT), 
    colors=(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, GL_COLOR_ATTACHMENT0)
)
class FBO(object):
    def __init__(self, attachments, size, ncolors=1):
        self.colors = []
        self.types = []

        fbo = glGenFramebuffers(1)
        #Bind the texture to the renderer's framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)

        if "colors" in attachments:
            for i in range(ncolors):
                tex = self._maketex("colors", size)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+i, GL_TEXTURE_2D, tex, 0)
                self.colors.append(tex)
                self.types.append(GL_COLOR_ATTACHMENT0+i)
            attachments.remove("colors")
        
        for kind in attachments:
            texform, textype, dtype, fbtype = fbotypes[kind]
            tex = self._maketex(kind, size=size)
            glFramebufferTexture2D(GL_FRAMEBUFFER, fbtype, GL_TEXTURE_2D, tex, 0)
            self.types.append(fbtype)
        
        #We always need a depth buffer! Otherwise occlusion will be messed up
        if "depth" not in attachments:
            rb = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, rb)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, size[0], size[1])
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.fbo = fbo
    
    def _maketex(self, kind, size, 
        magfilt=GL_NEAREST, minfilt=GL_NEAREST, 
        wrap_x=GL_CLAMP, wrap_y=GL_CLAMP):

        assert kind in fbotypes
        texform, textype, dtype, fbtype = fbotypes[kind]

        #First, create a texture
        frametex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, frametex)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexImage2D(GL_TEXTURE_2D, 0, texform, size[0], size[1], 0, textype, dtype, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        return frametex

class SSAOrender(object):
    def __init__(self, shaders, programs, win_size):
        self.renderer = Renderer(shaders, programs)
        self.renderer.add_shader("fsquad", GL_VERTEX_SHADER, "fsquad.v.glsl")
        self.renderer.add_shader("ssao_pass1", GL_FRAGMENT_SHADER, "ssao_pass1.f.glsl", "phong.f.glsl")
        self.renderer.add_shader("ssao_pass2", GL_FRAGMENT_SHADER, "ssao_pass2.f.glsl", "phong.f.glsl")

        #override the default shader with this passthru + ssao_pass1 to store depth
        self.renderer.add_program("ssao_pass1", ("passthru", "ssao_pass1"))
        self.renderer.add_program("ssao_pass2", ("fsquad", "ssao_pass2"))
        
        self.fbo = FBO(["colors"], ncolors=2, size=(win_size[0] / 2, win_size[1] / 2))
        randtex = np.random.rand(3, 128, 128)
        randtex = randtex.sum(-1)
        self.rnm = Texture(randtex.T)
        self.rnm.init()

    def draw(self, root, **kwargs):
        self.renderer.draw_to_fbo(self.fbo, root, shader="ssao_pass1", **kwargs)
        self.renderer.draw_fsquad("ssao_pass2", colors=self.fbo.colors[0], normalMap=self.fbo.colors[1], rnm=self.rnm.tex)


def test():
    import pygame
    pygame.init()
    pygame.display.set_mode((100,100), pygame.OPENGL | pygame.DOUBLEBUF)

    return Renderer(
                shaders=dict(
                    passthru=(GL_VERTEX_SHADER, "passthrough.v.glsl"),
                    phong=(GL_FRAGMENT_SHADER, "default.f.glsl", "phong.f.glsl")), 
                programs=dict(
                    default=("passthru", "phong"),
                )
            )