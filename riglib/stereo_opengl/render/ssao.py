import numpy as np
from OpenGL.GL import *

from render import Renderer
from fbo import FBOrender, FBO
from models import Texture
from utils import inspect_tex

class SSAO(FBOrender):
    def __init__(self, *args, **kwargs):
        super(SSAO, self).__init__(*args, **kwargs)
        self.sf = 3
        w, h = self.size[0] / self.sf, self.size[1] / self.sf
        self.normdepth = FBO(["colors", "depth"], size=(w,h))
        self.ping = FBO(['colors'], size=(w,h))
        self.pong = FBO(["colors"], size=(w,h))

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
        self.rnm = Texture(randtex.T, wrap_x=GL_REPEAT, wrap_y=GL_REPEAT, 
            magfilter=GL_NEAREST, minfilter=GL_NEAREST)
        self.rnm.init()

        self.clips = args[2], args[3]

    def draw(self, root, **kwargs):
        #First, draw the whole damned scene, but only read the normals and depth into ssao
        glViewport( 0,0, self.size[0]/self.sf, self.size[1]/self.sf)
        self.draw_to_fbo(self.normdepth, root, shader="ssao_pass1", **kwargs)
        
        #Now, do the actual ssao calculations, and draw it into ping
        self.draw_fsquad_to_fbo(self.pong, "ssao_pass2", 
            nearclip=self.clips[0], farclip=self.clips[1], 
            normalMap=self.normdepth.texs['colors'][0], depthMap=self.normdepth.texs['depth'], 
            rnm=self.rnm)
        
        #Reset the texture, draw into ping with hblur
        self.draw_fsquad_to_fbo(self.ping, "hblur", tex=self.pong.texs['colors'][0])
        #Reset the texture draw into pong with vblur
        self.draw_fsquad_to_fbo(self.pong, "vblur", tex=self.ping.texs['colors'][0])

        #Actually draw the final image to the screen
        glViewport(self.drawpos[0], self.drawpos[1], self.size[0], self.size[1])
        super(SSAO, self).draw(root, shader="ssao_pass3", shadow=self.pong.texs['colors'][0], 
            window=[float(self.drawpos[0]), self.drawpos[1], self.size[0], self.size[1]], **kwargs)
        
        self.draw_done()
    
    def draw_done(self):
        super(SSAO, self).draw_done()
        self.normdepth.clear()
        self.ping.clear()
        self.pong.clear()
