import numpy as np
from OpenGL.GL import *

from render import Renderer
from fbo import FBOrender

class SSAO(FBOrender):
    def __init__(self, *args, **kwargs):
        super(SSAO, self).__init__(*args, **kwargs)
        self.add_shader("fsquad", GL_VERTEX_SHADER, "fsquad.v.glsl")
        self.add_shader("ssao_pass1", GL_FRAGMENT_SHADER, "ssao_pass1.f.glsl", "phong.f.glsl")
        self.add_shader("ssao_pass2", GL_FRAGMENT_SHADER, "ssao_pass2.f.glsl")

        #override the default shader with this passthru + ssao_pass1 to store depth
        self.add_program("ssao_pass1", ("passthru", "ssao_pass1"))
        self.add_program("ssao_pass2", ("fsquad", "ssao_pass2"))
        
        self.fbo = FBO(["colors"], ncolors=2, size=(win_size[0] / 2, win_size[1] / 2))

        randtex = np.random.rand(3, 128, 128)
        randtex /= randtex.sum(0)
        self.rnm = Texture(randtex.T)
        self.rnm.init()
        self.renderer.get_texunit(self.rnm)
        self.renderer.get_texunit(self.fbo.colors[0])
        self.renderer.get_texunit(self.fbo.colors[1])
        
        self.clips = args[2], args[3]

    def draw(self, root, **kwargs):
        glClear(GL_COLOR_BUFFER_BIT)
        self.renderer.draw_to_fbo(self.fbo, root, shader="ssao_pass1",
            nearclip=self.clips[0], farclip=self.clips[1], **kwargs)
        
        ci, tu = self.renderer.get_texunit(self.fbo.colors[0])
        ni, tu = self.renderer.get_texunit(self.fbo.colors[1])
        rnm, tu = self.renderer.get_texunit(self.rnm)
        self.renderer.draw_fsquad("ssao_pass2", colors=ci, normalMap=ni, rnm=rnm)
