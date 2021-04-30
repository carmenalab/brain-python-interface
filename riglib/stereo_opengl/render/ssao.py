'''Needs docs'''

import numpy as np
from OpenGL.GL import *

from .render import Renderer
from .fbo import FBOrender, FBO
from ..textures import Texture

class SSAO(FBOrender):
    def __init__(self, *args, **kwargs):
        super(SSAO, self).__init__(*args, **kwargs)
        self.sf = 3
        w, h = self.size[0] / self.sf, self.size[1] / self.sf
        
        self.normdepth = FBO(["color0", "depth"], size=(w,h))
        self.ping = FBO(['color0'], size=(w,h))
        self.pong = FBO(["color0"], size=(w,h))

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

        randtex = np.random.rand(3, w, h)
        randtex /= randtex.sum(0)
        self.rnm = Texture(randtex.T, wrap_x=GL_REPEAT, wrap_y=GL_REPEAT, 
            magfilter=GL_NEAREST, minfilter=GL_NEAREST)
        self.rnm.init()

        self.clips = args[2], args[3]

    def draw(self, root, **kwargs):
        #First, draw the whole damned scene, but only read the normals and depth into ssao
        glPushAttrib(GL_VIEWPORT_BIT)
        glViewport( 0,0, self.size[0]/self.sf, self.size[1]/self.sf)
        self.draw_to_fbo(self.normdepth, root, shader="ssao_pass1", **kwargs)
        
        #Now, do the actual ssao calculations, and draw it into ping
        self.draw_fsquad_to_fbo(self.pong, "ssao_pass2", rnm=self.rnm,
            normalMap=self.normdepth['color0'], depthMap=self.normdepth['depth'],
            nearclip=self.clips[0], farclip=self.clips[1] )
        
        #Blur the textures
        self.draw_fsquad_to_fbo(self.ping, "hblur", tex=self.pong['color0'], blur=1./(self.size[0]/self.sf))
        self.draw_fsquad_to_fbo(self.pong, "vblur", tex=self.ping['color0'], blur=1./(self.size[0]/self.sf))
        
        glPopAttrib()
        #Actually draw the final image to the screen
        win = glGetIntegerv(GL_VIEWPORT)
        #Why is this call necessary at all?!
        glViewport(*win)
        
        super(SSAO, self).draw(root, shader="ssao_pass3", shadow=self.pong['color0'], 
            window=[float(i) for i in win], **kwargs)
        
        #self.draw_done()
    
    def clear(self):
        self.normdepth.clear()
        self.ping.clear()
        self.pong.clear()
    
    def draw_done(self):
        super(SSAO, self).draw_done()
        self.clear()
