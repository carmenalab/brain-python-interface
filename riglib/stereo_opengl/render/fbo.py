'''Needs docs'''


import numpy as np
from OpenGL.GL import *

from .render import Renderer
from ..textures import Texture

fbotypes = dict(
    depth=(GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT, GL_DEPTH_ATTACHMENT), 
    stencil=(GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, GL_STENCIL_ATTACHMENT), 
    colors=(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, GL_COLOR_ATTACHMENT0)
)

class FBO(object):
    def __init__(self, attachments, size=None, ncolors=1, **kwargs):
        maxcolors = range(glGetInteger(GL_MAX_COLOR_ATTACHMENTS))
        self.names = dict(("color%d"%i, GL_COLOR_ATTACHMENT0+i) for i in maxcolors)
        self.names["depth"] = GL_DEPTH_ATTACHMENT
        self.names["stencil"] = GL_STENCIL_ATTACHMENT

        self.textures = dict()

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        
        for attach in attachments:
            if isinstance(attach, str):
                if attach.startswith("color"):
                    idx = int(attach[5:])
                    attach = "colors"
                iform, exform, dtype, attachment = fbotypes[attach]
                texture = Texture(None, size=size, iformat=iform, exformat=exform, dtype=dtype)
                texture.init()
                if attach == "colors":
                    attachment += idx
                glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.tex, 0)
                self.textures[attachment] = texture
            else:
                attachment, texture = attach
                if attachment in self.names:
                    attachment = self.names[attachment]
                if texture is None and attachment == GL_DEPTH_ATTACHMENT:
                    rb = glGenRenderbuffers(1)
                    glBindRenderbuffer(GL_RENDERBUFFER, rb)
                    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, size[0], size[1])
                    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb)
                else:
                    if texture.tex is None:
                        texture.init()
                    glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.tex, 0)
                    self.textures[attachment] = texture
                
        types = [t for t in list(self.textures.keys()) if 
            t !=GL_DEPTH_ATTACHMENT and 
            t != GL_STENCIL_ATTACHMENT and 
            t != GL_DEPTH_STENCIL_ATTACHMENT]
        if len(types) > 0:
            glDrawBuffers(types)
        else:
            glDrawBuffers(GL_NONE)
        
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.textures[self.names[idx]]
        
        return self.textures[idx]
    
    def clear(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

class FBOrender(Renderer):
    def draw_fsquad(self, shader, **kwargs):
        ctx = self.programs[shader]
        glUseProgram(ctx.program)
        for name, v in list(kwargs.items()):
            if isinstance(v, Texture):
                ctx.uniforms[name] = self.get_texunit(v)
            else:
                ctx.uniforms[name] = v
        
        glEnableVertexAttribArray(ctx.attributes['position'])
        glBindBuffer(GL_ARRAY_BUFFER, self.fsquad_buf[0])
        glVertexAttribPointer(ctx.attributes['position'],
            4, GL_FLOAT, GL_FALSE, 4*4, GLvoidp(0))
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.fsquad_buf[1]);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, GLvoidp(0))
        glDisableVertexAttribArray(ctx.attributes['position'])
    
    def draw_fsquad_to_fbo(self, fbo, shader, **kwargs):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        self.draw_fsquad(shader, **kwargs)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def draw_to_fbo(self, fbo, root, **kwargs):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo)
        #Erase old buffer info
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        super(FBOrender, self).draw(root, **kwargs)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    