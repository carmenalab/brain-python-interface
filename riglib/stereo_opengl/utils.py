from __future__ import division
import numpy as np

def frustum(l, r, t, b, n, f):
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
    raise ValueError("This function is incorrect! Go find your own perspective matrix!")
    f = 1./ np.tan(np.radians(angle))
    fn, nfn = far + near, far - near
    return np.array([[f/aspect, 0,    0,      0],
                     [0,        f,    0,      0],
                     [0,        0, fn/nfn, 2*far*near/nfn],
                     [0,        0,   -1,      0]])

def offaxis_frusta(winsize, fov, near, far, focal_dist, iod):
    aspect = winsize[0] / winsize[1]
    top = near * np.tan(np.radians(fov) / 2)
    right = aspect*top
    fshift = 0.5*iod*near / focal_dist

    left = frustum(-right+fshift, right+fshift, top, -top, near, far)
    right = frustum(-right-fshift, right-fshift, top, -top, near, far)
    return left, right


def cloudy_tex(size=(512,512)):
    im = np.random.randn(*size)
    grid = np.mgrid[-1:1:size[0]*1j, -1:1:size[1]*1j]
    mask = 1/(grid**2).sum(0)
    fim = np.fft.fftshift(np.fft.fft2(im))
    return np.abs(np.fft.ifft2(np.fft.fftshift(mask * fim)))

def makeSphere(radius, segments=36):
    #triangles = np.zeros((?,3))
    zvals = radius * np.cos(np.linspace(0, np.pi, num=segments))
    circlevals = np.linspace(0, 2*pi, num=segments, endpoint=False)

    vertices = np.zeros(((len(zvals)-2) * len(circlevals), 3))

    for i, z in enumerate(zvals[1:-1]):
        circlepoints = np.zeros((segments, 3))
        circlepoints[:,2] = z
        r = np.sqrt(radius**2 - z**2)
        circlepoints[:,0] = r*np.sin(circlevals)
        circlepoints[:,1] = r*np.cos(circlevals)
        vertices[segments*i:segments*(i+1),:] = circlepoints
    
    vertices = np.vstack([(0,0,1), vertices,(0,0,-1)])

    polys = []
    for j in range(segments-2):
        j *= segments
        for i in range(segments-1):
            polys.append((i+j, i+j+1, i+j+segments))
            polys.append((i+j+segments, i+j+1, i+j+1+segments))
        polys.append((segments+j-1, j, segments*2-1+j))
        polys.append((segments+j*2-1, j, segments+j))

    return vertices, np.array(polys)
