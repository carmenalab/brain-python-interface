
'''Needs docs'''
import numpy as np
from .textures import Texture

def frustum(l, r, t, b, n, f):
    '''
    This function emulates glFrustum: https://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml
    A frustum is a solid cut by planes, e.g., the planes representing the viewable area of a screen.

    Parameters
    ----------
    l: float
        Distance to the left plane of the screen
    r: float
        Distance to the right plane of the screen
    t: float
        Distance to the top plane of the screen
    b: float
        Distance to the bottom plane of the screen
    n: float
        Distance to the near plane of the screen
    f: float
        Distance to the far plane of the screen

    Returns
    -------
    Projection matrix to apply to solid to truncate

    '''
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
    fn = far + near
    nfn = far - near
    return np.array([[f/aspect, 0,    0,               0],
                     [0,        f,    0,               0],
                     [0,        0, -fn/nfn, -2*far*near/nfn],
                     [0,        0,   -1,               0]])

def orthographic(w, h, near, far):
    fn = far + near
    nfn = far - near
    return np.array([[2/w, 0,   0,      0],
                     [0,   2/h, 0,      0],
                     [0,   0,   -2/nfn, -fn/nfn],
                     [0,   0,   0,       1]])

def offaxis_frusta(winsize, fov, near, far, focal_dist, iod, flip=False, flip_z=False):
    aspect = winsize[0] / winsize[1]
    top = near * np.tan(np.radians(fov) / 2)
    right = aspect*top
    fshift = (iod/2) * near / focal_dist

    # calculate the perspective matrix for the left eye and for the right eye
    left = frustum(-right+fshift, right+fshift, top, -top, near, far)
    right = frustum(-right-fshift, right-fshift, top, -top, near, far)
    
    # multiply in the iod (intraocular distance) modelview transform
    lxfm, rxfm = np.eye(4), np.eye(4)
    lxfm[:3,-1] = [0.5*iod, 0, 0]
    rxfm[:3,-1] = [-0.5*iod, 0, 0]
    flip_mat = np.eye(4)


    if flip:
        flip_mat[0,0] = -1
    if flip_z:
        flip_mat[1,1] = -1

    return np.dot(flip_mat, np.dot(left, lxfm)), np.dot(flip_mat, np.dot(right, rxfm))

    #return np.dot(left, lxfm), np.dot(right, rxfm)

def cloudy_tex(size=(512,512)):
    '''Generates 1/f distributed noise and puts it into a texture. Looks like clouds'''
    im = np.random.randn(*size)
    grid = np.mgrid[-1:1:size[0]*1j, -1:1:size[1]*1j]
    mask = 1/(grid**2).sum(0)
    fim = np.fft.fftshift(np.fft.fft2(im))
    im = np.abs(np.fft.ifft2(np.fft.fftshift(mask * fim)))
    im -= im.min()
    return Texture(im / im.max())
