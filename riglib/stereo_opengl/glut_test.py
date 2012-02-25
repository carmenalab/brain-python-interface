from __future__ import division

import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *

vshade = '''
#version 110

uniform mat4 p_matrix;
uniform mat4 xfm;

attribute vec4 position;

void main(void)
{
    vec4 eye_position = xfm * position;
    gl_Position = p_matrix * eye_position;
}
'''

fshade = '''
#version 120

uniform vec4 basecolor;

void main() {
    gl_FragColor = basecolor;
}
'''

def _make_shader(stype, src):
    shader = glCreateShader(stype)
    glShaderSource(shader, src)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        err = glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        raise Exception(err)
    
    return shader

def _make_poly(verts, polys):
    verts, polys = np.array(verts), np.array(polys)
    vbuf = glGenBuffers(1)
    ebuf = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbuf)
    glBufferData(GL_ARRAY_BUFFER, 
        verts.astype(np.float32).ravel(), GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebuf)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
        polys.astype(np.uint16).ravel(), GL_STATIC_DRAW)
    return vbuf, ebuf

def _draw_plane(ploc, vbuf, ebuf):
    glEnableVertexAttribArray(ploc)
    glBindBuffer(GL_ARRAY_BUFFER, vbuf)
    glVertexAttribPointer(ploc, 4, GL_FLOAT, GL_FALSE, 4*4, GLvoidp(0))

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebuf);
    glDrawElements(GL_TRIANGLES, 2*3, GL_UNSIGNED_SHORT, GLvoidp(0))
    glDisableVertexAttribArray(ploc)
    
def perspective(angle, aspect, near, far):
    '''Generates a perspective transform matrix'''
    f = 1./ np.tan(np.radians(angle))
    fn, nfn = far + near, far - near
    return np.array([[f/aspect, 0,    0,      0],
                     [0,        f,    0,      0],
                     [0,        0, fn/nfn, 2*far*near/nfn],
                     [0,        0,   -1,      0]])

class Window(object):
    window_size = (640, 480)

    background = (0,0,0,1)
    fps = 60
    fov = 60

    def __init__(self):
        super(Window, self).__init__()
        self.models = []
        self.eyepos = [0,-2,0]

    def init(self):
        glEnable(GL_BLEND)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST) 
        glEnable(GL_TEXTURE_2D)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*self.background)
        glClearDepth(1.0)
        
        self.vshade = _make_shader(GL_VERTEX_SHADER, vshade)
        self.fshade = _make_shader(GL_FRAGMENT_SHADER, fshade)
        self.program = glCreateProgram()
        glAttachShader(self.program, self.vshade)
        glAttachShader(self.program, self.fshade)
        glLinkProgram(self.program)
        
        w, h = self.window_size
        self.projection = perspective(self.fov/2, w/h, 0.0625, 256.).astype(np.float32).ravel()

        self.mv = np.eye(4)
        self.mv[:3,-1] = [0,0,-5]
        self.mv = self.mv.astype(np.float32).ravel()

        self.plane1 = [(0,0,1,1),(1,0,1,1),(1,1,1,1),(0,1,1,1)], [(0,1,3), (3,1,2)]
        self.plane2 = [(0,0,-4,1),(1,0,-4,1),(1,1,-4,1),(0,1,-4,1)], [(0,1,3), (3,1,2)]
        self.plane1 = _make_poly(*self.plane1)
        self.plane2 = _make_poly(*self.plane2)

        self.ploc = glGetUniformLocation(self.program, "p_matrix")
        self.mvloc = glGetUniformLocation(self.program, "xfm")
        self.cloc = glGetUniformLocation(self.program, "basecolor")
        self.vloc = glGetAttribLocation(self.program, "position")
    
    def render(self):
        w, h = self.window_size
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, w, h)
        glUseProgram(self.program)
        glUniformMatrix4fv(self.ploc, 1, GL_TRUE, self.projection)
        glUniformMatrix4fv(self.mvloc, 1, GL_TRUE, self.mv)

        glUniform4f(self.cloc, 1, 0, 0, 1)
        _draw_plane(self.vloc, *self.plane1)
        glUniform4f(self.cloc, 0, 0, 1, 1)
        _draw_plane(self.vloc, *self.plane2)

        glutSwapBuffers()

def main():
    win = Window()
    glutInit();
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(*win.window_size);
    glutCreateWindow("Test");
    glutDisplayFunc(win.render);
    win.init()

    glutMainLoop()

if __name__ == "__main__":
    main()