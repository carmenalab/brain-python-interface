from __future__ import division
import numpy as np
from textures import Texture
from OpenGL.GL import glBindTexture, glGetTexImage, GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE

def frustum(l, r, t, b, n, f):
    '''Emulates glFrustum'''
    rl, nrl = r + l, r - l
    tb, ntb = t + b, t - b
    fn, nfn = f + n, f - n
    return np.array([[2*n / nrl,     0,     rl / nrl,       0],
                     [    0,     2*n / ntb, tb / ntb,       0],
                     [    0,         0,    -fn / nfn, -2*f*n / nfn],
                     [    0,         0,       -1,           0]])

def perspective(angle, aspect, near, far):
    '''Generates a perspective transform matrix'''
    f = 1./ np.tan(np.radians(angle) / 2)
    fn, nfn = far + near, far - near
    return np.array([[f/aspect, 0,    0,               0],
                     [0,        f,    0,               0],
                     [0,        0, -fn/nfn, -2*far*near/nfn],
                     [0,        0,   -1,               0]])

def offaxis_frusta(winsize, fov, near, far, focal_dist, iod):
    aspect = winsize[0] / winsize[1]
    top = near * np.tan(np.radians(fov) / 2)
    right = aspect*top
    fshift = (iod/2) * near / focal_dist

    #multiply in the iod modelview transform
    lxfm, rxfm = np.eye(4), np.eye(4)
    lxfm[:3,-1] = [0.5*iod, 0, 0]
    rxfm[:3,-1] = [-0.5*iod, 0, 0]

    left = frustum(-right+fshift, right+fshift, top, -top, near, far)
    right = frustum(-right-fshift, right-fshift, top, -top, near, far)
    return np.dot(left, lxfm), np.dot(right, rxfm)

def mirror_frusta(winsize, fov, near, far, focal_dist, iod):
    aspect = winsize[0] / winsize[1]
    top = near * np.tan(np.radians(fov) / 2)
    right = aspect*top

    #multiply in the iod modelview transform
    lxfm, rxfm = np.eye(4), np.eye(4)
    lxfm[:3,-1] = [0.5*iod, 0, 0]
    lxfm[0,0] = -1
    rxfm[:3,-1] = [-0.5*iod, 0, 0]
    rxfm[0,0] = -1

    left = frustum(-right, right, top, -top, near, far)
    right = frustum(-right, right, top, -top, near, far)
    return np.dot(left, lxfm), np.dot(right, rxfm)

def cloudy_tex(size=(512,512)):
    im = np.random.randn(*size)
    grid = np.mgrid[-1:1:size[0]*1j, -1:1:size[1]*1j]
    mask = 1/(grid**2).sum(0)
    fim = np.fft.fftshift(np.fft.fft2(im))
    im = np.abs(np.fft.ifft2(np.fft.fftshift(mask * fim)))
    im -= im.min()
    return Texture(im / im.max())

class Quaternion(object):
    def __init__(self, w=1, x=0, y=0, z=0):
        self.quat = np.array([w, x, y, z])
    
    def __str__(self):
        return "%f+%fi+%fj+%fk"%self.quat
    
    def norm(self):
        self.quat /= np.sqrt((self.quat**2).sum())
    
    def conj(self):
        return Quaternion(w, -x, -y, -z)
    
    def __getattr__(self, attr):
        if attr in ["w", "scalar"]:
            return self.quat[0]
        elif attr in ["x", "i"]:
            return self.quat[1]
        elif attr in ["y", "j"]:
            return self.quat[2]
        elif attr in ["z", "k"]:
            return self.quat[3]
        elif attr in ["v", "vec", "vector"]:
            return self.quat[1:]
        else:
            super(Quaternion, self).__getattr__(self, attr)
    
    def __mult__(self, other):
        if isinstance(other, Quaternion):
            w = self.w*other.w   - np.dot(self.vec, other.vec)
            v = self.w*other.vec + other.w*self.vec + np.cross(self.vec, other.vec)
            return Quaternion(w, *v)
        elif isinstance(other, np.ndarray):
            #rotate a vector, will need to be implemented in GLSL eventually
            conj = self.conj
            w = -np.dot(other, conj.vec)
            vec = conj.w*other + np.cross(other, conj.vec)
            nw = self.w*w - np.dot(self.vec, vec)
            pts = self.w*vec + w*self.vec + np.cross(self.vec, vec)
            return nw, pts
    
    @classmethod
    def from_axis(cls, axis, angle):
        return cls(w, *v)

class Transform(object):
    def __init__(self, move=(0,0,0), scale=1, rotate=None):
        self.move = move
        self.scale = 1
        self.rotate = rotate if rotate is not None else Quaternion()
    
    def __mult__(self, other):
        move = self.move + other.move
        scale = self.scale * other.scale
        rot = self.rotate * other.rotate
        return Transform(move, scale, rot)
    
    def to_mat(self):
        pass
