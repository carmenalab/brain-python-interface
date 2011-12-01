import numpy as np

def checkerboard(size=(500,500), n=6):
    square = np.ones(np.array(size) / n)
    line = np.hstack([np.hstack([square, 0*square])]*(n/2))
    return np.vstack([line,line[:,::-1]]*(n/2)).astype(bool)

def squaremask(size=(500,500), square=200):
    data = np.zeros(size, dtype=bool)
    top, bottom = size[0]/2-square/2, size[0]/2+square/2
    left, right = size[1]/2-square/2, size[1]/2+square/2
    data[top:bottom, left:right] = True
    return data

def generate(mask, offset=10):
    data = np.random.randn(*mask.shape)
    left, right = data.copy(), data.copy()
    leftmask = np.roll(mask, offset, axis=1)
    rightmask = np.roll(mask, -offset, axis=1)
    left[mask] = data[leftmask]
    right[mask] = data[rightmask]
    return left, right, data

def compile(mask, offset=10, size=(1080, 3840)):
    left, right, flat = generate(mask, offset)
    
    top, bottom = map(lambda x: size[0] / 2 + x*left.shape[0] / 2, [-1, 1])
    lleft, lright = map(lambda x: size[1] / 4 + x*left.shape[1] / 2, [-1, 1])
    rleft, rright = map(lambda x: 3*size[1] / 4 + x*left.shape[1] / 2, [-1, 1])
    
    img = np.zeros(size)
    img[top:bottom, lleft:lright] = left
    img[top:bottom, rleft:rright] = right
    
    flatim = np.zeros(size)
    flatim[top:bottom, lleft:lright] = flat
    flatim[top:bottom, rleft:rright] = flat
    
    return img, flatim
    
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    if len(sys.argv) < 2:
        mask = squaremask()
    else:
        mask = plt.imread(sys.argv[1]).mean(-1) != 0
        
    stereo, flat = compile(mask)
    plt.imsave("stereo.png", stereo > 0, cmap=cm.Greys_r)
    plt.imsave("flat.png", flat > 0, cmap=cm.Greys_r)
