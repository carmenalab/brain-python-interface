import numpy as np

def angle_inside_range(angle, start, end):
    '''Test whether angle in inside range [start, end).'''

    # ensure that end value is within the range [start, start + 2*pi)
    while end < start:
        end += 2*np.pi
    while end >= start + 2*np.pi:
        end -= 2*np.pi

    # add/subtract 2*pi until angle is within the range [start, start + 2*pi)
    while angle < start:
        angle += 2*np.pi
    while angle >= start + 2*np.pi:
        angle -= 2*np.pi

    return (angle >= start) and (angle < end)

def angle_subtract(angle1, angle2):
    '''Compute angle1 minus angle2 so that result is in range [-pi, pi).'''
    
    result = angle1 - angle2
    while result < -np.pi:
        result += 2*np.pi
    while result >= np.pi:
        result -= 2*np.pi

    return result

def angle_subtract_vec(angle_vec1, angle_vec2):
    result = np.zeros(angle_vec1.shape)

    for i in range(len(result)):
        result[i] = angle_subtract(angle_vec1[i], angle_vec2[i])

    return result

def l1_ang_dist(a, b):
    '''Calculate the l1 distance between angular vectors a and b.'''

    return sum(abs(angle_subtract(ai, bi)) for (ai, bi) in zip(a, b))
