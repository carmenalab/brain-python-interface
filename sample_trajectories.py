import scipy.io as sio
import numpy as np

from riglib.ismore import ArmAssistData, ReHandData

filename = '/Users/sdangi/Desktop/Kinematic data ArmAssist/Epoched data/epoched_kin_data/NI_sess05_20140610/NI_B1S005R01.mat'

mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

kin_epoched = mat['kin_epoched']

print ArmAssistData.dtype
