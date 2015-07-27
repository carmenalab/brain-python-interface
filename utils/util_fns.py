import numpy as np

def dist(a, b):
    '''Calculate the Euclidean distance between vectors a and b.'''

    return np.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

def norm_vec(x, eps=1e-9):
    return x / (np.linalg.norm(x) + eps)

def bound(x, x_min, x_max):
    if x_min > x_max:
        raise Exception('Error in bound function: min bound is greater than max bound!')
    
    x_bounded = x
    if not np.isnan(x_min):
        x_bounded = max(x_bounded, x_min)
    if not np.isnan(x_max):
        x_bounded = min(x_bounded, x_max)

    return x_bounded