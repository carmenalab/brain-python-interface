import tables
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import pickle


from ismore import brainamp_channel_lists
from ismore.emg_decoding import LinearEMGDecoder
from ismore.emg_feature_extraction import EMGMultiFeatureExtractor
from ismore.common_state_lists import *
from utils.constants import *


################################################################
# The variables below need to be set before running this script!
################################################################

# Set 'pkl_name' to be the full name of the .pkl file in which the trained
#   decoder will be saved
pkl_name = '/storage/rawdata/bmi/emg_decoder_1.pkl'

# Set 'okay_to_overwrite_decoder_file' to True to allow overwriting an existing 
#   file
okay_to_overwrite_decoder_file = True

# Set 'train_hdf_names' to be a list of names of .hdf files to train from
train_hdf_names = [
    #'/home/lab/Desktop/test20150424_04.hdf',
    #'/home/lab/Desktop/AS_F1S001R02.hdf',
    #'/home/lab/Desktop/AS_F2S001R02.hdf'
   # '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/AS_F1S001R02.hdf'
   # '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/AS_F2S001R02.hdf'
    '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_06.hdf', #B1_R1
    '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_07.hdf', #B1_R2
    '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_08.hdf', #B1_R3
    '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_09.hdf', #B1_R4
    #'/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_11.hdf' #B1_R5
]

# Set 'test_hdf_names' to be a list of names of .hdf files to test on (offline)
test_hdf_names = [
    '/home/tecnalia/hybries/resources/EMGdecoder_code_testing/fs20150424_11.hdf' #B1_R5   
]

# Set 'channels' to be a list of channel names
channels = brainamp_channel_lists.emg

# Set 'plant_type' (type of plant for which to create a decoder)
#   choices: 'ArmAssist', 'ReHand', or 'IsMore'
plant_type = 'IsMore'

# Set 'feature_names' to be a list containing the names of features to use
#   (see emg_feature_extraction.py for options)
feature_names = ['MAV', 'WAMP', 'VAR', 'WL', 'RMS', 'ZC', 'SSC']

# Set 'feature_fn_kwargs' to be a dictionary where:
#   key: name of a feature (e.g., 'ZC')
#   value: a dictionary of the keyword arguments to be passed into the feature
#          function (e.g., extract_ZC) corresponding to this feature
#   (see emg_feature_extraction.py for how the feature function definitions)
feature_fn_kwargs = {
    'WAMP': {'threshold': 30},  
    'ZC':   {'threshold': 30},
    'SSC':  {'threshold': 700},
}

# Set 'win_len'
win_len = 0.2  # secs

# Set 'fs'
fs = 1000  # Hz

# Set 'K' to be the value of the ridge parameter
K = 0  # 10e4

# Set 'states_to_flip' to be a list containing the state names for which
#   the beta vector of coefficients (trained using ridge regression) should be 
#   flipped (i.e., multiplied by -1)
states_to_flip = ['aa_vx', 'aa_vpsi', 'rh_vprono']

########################################################

if os.path.isfile(pkl_name) and not okay_to_overwrite_decoder_file:
    raise Exception('A decoder file with that name already exists!') 

extractor_cls = EMGMultiFeatureExtractor
extractor_kwargs = {
    'channels':          channels,
    'feature_names':     feature_names,
    'feature_fn_kwargs': feature_fn_kwargs,
    'win_len':           win_len,
    'fs':                fs,
}

decoder = LinearEMGDecoder(channels, plant_type, fs, win_len, extractor_cls, extractor_kwargs)
decoder.train_ridge(K, train_hdf_names, test_hdf_names, states_to_flip)

pickle.dump(decoder, open(pkl_name, 'wb'))
