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
