import numpy as np
from OpenGL.GL import *

from render import Renderer
from fbo import FBOrender, FBO
from models import Texture

class SSAO(FBOrender):
    def __init__(self, window_size, *args, **kwargs):
        self.ssao = FBO(["colors", "depth"], size=(window_size[0] / 2, window_size[1] / 2))
        self.ping = FBO(['colors'], size=(window_size[0] / 2, window_size[1] / 2))
        self.pong = FBO(["colors"], size=(window_size[0] / 2, window_size[1] / 2))
        super(SSAO, self).__init__(window_size, *args, **kwargs)

        self.add_shader("fsquad", GL_VERTEX_SHADER, "fsquad.v.glsl")
        self.add_shader("ssao_pass1", GL_FRAGMENT_SHADER, "ssao_pass1.f.glsl")
        self.add_shader("ssao_pass2", GL_FRAGMENT_SHADER, "ssao_pass2.f.glsl")
        self.add_shader("ssao_pass3", GL_FRAGMENT_SHADER, "ssao_pass3.f.glsl", "phong.f.glsl")
        self.add_shader("hblur", GL_FRAGMENT_SHADER, "hblur.f.glsl")
        self.add_shader("vblur", GL_FRAGMENT_SHADER, "vblur.f.glsl")

        #override the default shader with this passthru + ssao_pass1 to store depth
        self.add_program("ssao_pass1", ("passthru", "ssao_pass1"))
        self.add_program("ssao_pass2", ("fsquad", "ssao_pass2"))
        self.add_program("hblur", ("fsquad", "hblur"))
        self.add_program("vblur", ("fsquad", "vblur"))
        self.add_program("ssao_pass3", ("passthru", "ssao_pass3"))

        randtex = np.random.rand(3, 128, 128)
        randtex /= randtex.sum(0)
        self.rnm = Texture(randtex.T)
        self.rnm.init()
        self.get_texunit(self.rnm)
        self.get_texunit(self.ssao.texs['colors'][0])
        self.get_texunit(self.ssao.texs['depth'])
        self.get_texunit(self.ping.texs['colors'][0])
        self.get_texunit(self.pong.texs['colors'][0])

        self.clips = args[1], args[2]

    def draw(self, root, **kwargs):
        glClear(GL_COLOR_BUFFER_BIT)
        #First, draw the whole damned scene, but only read the normals and depth into ssao
        old_size = self.size
        self.size = self.size[0]/2, self.size[1]/2
        glViewport(0,0,self.size[0], self.size[1])
        self.draw_to_fbo(self.ssao, root, shader="ssao_pass1", **kwargs)
        
        #reset all the textures because opengl is stupid
        nm, tu = self.get_texunit(self.ssao.texs['colors'][0])
        dm, tu = self.get_texunit(self.ssao.texs['depth'])
        rnm, tu = self.get_texunit(self.rnm)
        ping, tu = self.get_texunit(self.ping.texs['colors'][0])
        pong, tu = self.get_texunit(self.pong.texs['colors'][0])
        #Why do I get unbound?
        self.ssao.texs['colors'][0].set(nm)
        self.ssao.texs['depth'].set(dm)

        #Now, do the actual ssao calculations, and raw it into ping
        self.draw_fsquad_to_fbo(self.pong, "ssao_pass2", 
            nearclip=self.clips[0], farclip=self.clips[1], 
            normalMap=nm, depthMap=dm, rnm=rnm)
        
        #Reset the texture, draw into ping with hblur
        self.pong.texs['colors'][0].set(pong)
        self.draw_fsquad_to_fbo(self.ping, "hblur", tex=pong)
        #Reset the texture draw into pong with vblur
        self.ping.texs['colors'][0].set(ping)
        self.draw_fsquad_to_fbo(self.pong, "vblur", tex=ping)
        
        #Now pong should contain the shadow
        self.pong.texs['colors'][0].set(pong)
        self.size = old_size
        glViewport(0,0,self.size[0], self.size[1])
        #np.save("/tmp/test.npy", np.fromstring(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE), dtype=np.uint8).reshape(-1,640,4))
        super(SSAO, self).draw(root, shader="ssao_pass3", shadow=pong, 
            width=float(self.size[0]), height=float(self.size[1]))

        