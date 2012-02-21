from OpenGL.GL import *

class Model(object):
    def __init__(self, xfm, verts, faces):
        self.xfm = xfm
        self.verts = verts
        self.faces = faces