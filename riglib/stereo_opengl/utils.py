from __future__ import division
import numpy as np

def frustum(l, r, u, b, n, f):
    '''Emulates glFrustum'''
    rl, nrl = r + l, r - l
    tb, ntb = t + b, t - b
    fn, nfn = f + n, f - n
    return np.array([[2*n / nrl, 0, rl / nrl, 0],
                     [0, 2*n / ntb, tb / ntb, 0],
                     [0,0,-fn / nfn, -2*f*n / nfn],
                     [0,0,-1,0]])

def perspective(angle, aspect, near, far):
    '''Generates a perspective transform matrix'''
    f = 1./ np.tan(np.radians(angle))
    fn, nfn = far + near, far - near
    return np.array([[f/aspect, 0,    0,      0],
                     [0,        f,    0,      0],
                     [0,        0, fn/nfn, 2*far*near/nfn],
                     [0,        0,   -1,      0]])

def offaxis_frustra(winsize, fov, near, far, focal_dist, iod):
    aspect = winsize[0] / winsize[1]
    wdiv2 = near * np.tan(np.radians(fov) / 2)

    t, b = wdiv2, -wdiv2
    l = -aspect * wdiv2 - 0.5*iod*near / focal_dist
    r = -aspect * wdiv2 + 0.5*iod*near / focal_dist

    left = frustrum(l, -l, t, b, near, far)
    right = frustrum(r, -r, t, b, near, far)


def cloudy_tex(size=(512,512)
    im = np.random.randn(*size)
    grid = np.mgrid[-1:1:size[0]*1j, -1:1:size[1]*1j]
    mask = 1/(grid**2).sum(0)
    fim = np.fft.fftshift(np.fft.fft2(im))
    return np.abs(np.fft.ifft2(np.fft.fftshift(mask * fim)))