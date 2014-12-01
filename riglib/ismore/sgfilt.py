import numpy as np

def compute_sgfilt_imp_resp(N, M_L, M_R):
    '''Compute Savitzky-Golay filter impulse response.'''

    tmp = np.arange(-M_L, M_R+1, dtype='double')
    a = np.polyfit(tmp, np.hstack([np.zeros(M_L), 1, np.zeros(M_R)]), N)
    h = np.polyval(a, tmp)[::-1]

    return h