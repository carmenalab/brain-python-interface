'''
To use on command line, cd into the directory containing this file and type:
    python test_KF_EMG_decoding.py <full path to .mat file>
e.g., 
    python test_KF_EMG_decoding.py /Users/sdangi/Dropbox/KF_data/dataset1.mat
'''


import scipy.io as sio
import numpy as np
from scipy.stats import pearsonr
import ntpath
import sys
import fnmatch

from riglib.bmi import bmi, kfdecoder

# name of .mat file is passed as first argument
matfile = sys.argv[1]

# load variables from .mat file
mat = sio.loadmat(matfile, struct_as_record=False, squeeze_me=True)
for var_name in list(mat.keys()):
    if fnmatch.fnmatch(var_name, 'emg*_train*'):
        emg_train = mat[var_name]
    elif fnmatch.fnmatch(var_name, 'kin*_train*'):
        kin_train = mat[var_name]
    elif fnmatch.fnmatch(var_name, 'emg*_test*'):
        emg_test = mat[var_name]
    elif fnmatch.fnmatch(var_name, 'kin*_test*'):
        kin_test = mat[var_name]

# fit KF matrices from training data
A, W = kfdecoder.KalmanFilter.MLE_state_space_model(kin_train, include_offset=True)
C, Q = kfdecoder.KalmanFilter.MLE_obs_model(kin_train, emg_train)

# create a KalmanFilter object
kf = kfdecoder.KalmanFilter(A, W, C, Q)

# calculate converged K and P matrices (without doing any KF predictions)
# this function will print out how many iterations it takes for the matrix norm
#   of the difference between K matrices on successive iterations to go below
#   the tolerance value 'tol'
F, K, P = kf.get_sskf(return_P=True, verbose=True, tol=1e-15)

# define the starting state of the KF to be zeros (with a 1 for the last entry)
len_x = kin_test.shape[0] + 1
x_mean = np.zeros((len_x, 1))
x_mean[-1] = 1
x_cov = np.zeros((len_x, len_x))
x = bmi.GaussianState(x_mean, x_cov)

# predictions will be stored in this matrixs
kin_pred = np.zeros((len_x, kin_test.shape[1]))

# iterate through each column of emg_test and predict the kinematics
for i, y in enumerate(emg_test.T):
    y = y.reshape(-1, 1)
    x = kf._forward_infer(x, y)
    kin_pred[:, i] = np.array(x.mean).reshape(-1)

# calculate the correlation coefficient for each state variable
cc_values = np.array([pearsonr(kin_test[i, :], kin_pred[i, :])[0] for i in range(len_x-1)])

print('datafile:', matfile)
print('cc values:', cc_values)

# save the results into a .mat file
vars_to_save = {
    'A': A,
    'W': W,
    'C': C,
    'Q': Q,
    'K': K,
    'P': P,
    'cc_values': cc_values,
}
sio.matlab.savemat(matfile[:-4] + '_pyresults.mat', vars_to_save)

