'''Utility functions.'''

import numpy as np
import multiprocessing as mp

def shared_np_mat(dtype, shape):
    '''Return a numpy matrix, with specified dtype and shape, that lives in
    shared memory so that it can be read/set by multiple processes.

    Note: when setting the value of a shared_np_mat variable, be sure
    to do "x[:, :] = ..." instead of "x = ...".  If you do the latter, you 
    are mistakenly re-assigning the variable name "x" to a new object (that
    probably doesn't live in shared memory, which is not what you want).
    '''
    tmp = mp.RawArray('c', dtype.itemsize * shape[0] * shape[1])
    return np.mat(np.frombuffer(tmp, dtype).reshape(shape))