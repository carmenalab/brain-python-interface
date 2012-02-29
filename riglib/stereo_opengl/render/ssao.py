import numpy as np
from OpenGL.GL import *

from render import Renderer
from fbo import FBOrender, FBO
from models import Texture

class SSAO(FBOrender):
    def __init__(self, window_size, *args, **kwargs):
        fbo = FBO(["colors"], size=(window_size[0] / 2, window_size[1] / 2))
        super(SSAO, self).__init__(fbo, window_size, *args, **kwargs)

        self.add_shader("fsquad", GL_VERTEX_SHADER, "fsquad.v.glsl")
        self.add_shader("ssao_pass1", GL_FRAGMENT_SHADER, "ssao_pass1.f.glsl", "phong.f.glsl")
        self.add_shader("ssao_pass2", GL_FRAGMENT_SHADER, "ssao_pass2.f.glsl")

        #override the default shader with this passthru + ssao_pass1 to store depth
        self.add_program("ssao_pass1", ("passthru", "ssao_pass1"))
        self.add_program("ssao_pass2", ("fsquad", "ssao_pass2"))

        randtex = np.random.rand(3, 128, 128)
        randtex /= randtex.sum(0)
        self.rnm = Texture(randtex.T)
        self.rnm.init()
        self.get_texunit(self.rnm)
        self.get_texunit(self.fbo.colors[0])

        self.clips = args[1], args[2]

    def draw(self, root, **kwargs):
        glClear(GL_COLOR_BUFFER_BIT)
        old_size = self.size
        self.size = self.size[0]/2, self.size[1]/2
        glViewport(0,0,self.size[0], self.size[1])
        self.draw_to_fbo(root, shader="ssao_pass1",
            nearclip=self.clips[0], farclip=self.clips[1], **kwargs)
        
        nm, tu = self.get_texunit(self.fbo.colors[0])
        rnm, tu = self.get_texunit(self.rnm)
        self.size = old_size
        glViewport(0,0,self.size[0], self.size[1])
        self.draw_fsquad("ssao_pass2", normalMap=nm, rnm=rnm)
