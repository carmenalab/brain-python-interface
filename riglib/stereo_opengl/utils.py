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