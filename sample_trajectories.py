import scipy.io as sio
import numpy as np
import pandas as pd
import tables
import pickle

from utils.constants import *


pkl_name = 'trajectories.pkl'
filename = '/home/lab/Desktop/Kinematic data ArmAssist/Epoched data/epoched_kin_data/NI_sess05_20140610/NI_B1S005R01.mat'

index = [
    'Time',
    'AbsPos_X(mm)',
    'AbsPos_Y(mm)',
    'AbsPos_Angle(dg)',
    'Kalman_X(mm)',
    'Kalman_Y(mm)',
    'Kalman_Angle(dg)',
    'Arm_Force(gr)',
    'Arm_Angle(dg)',
    'Supination(dg)',
    'Thumb(dg)',
    'Index(dg)',
    'Fingers(dg)',
]

field_mapping = {
    'Time':             'ts', 
    'AbsPos_X(mm)':     'aa_px',
    'AbsPos_Y(mm)':     'aa_py',
    'AbsPos_Angle(dg)': 'aa_ppsi',
    'Supination(dg)':   'rh_pprono',
    'Thumb(dg)':        'rh_pthumb',
    'Index(dg)':        'rh_pindex',
    'Fingers(dg)':      'rh_pfing3',
}



aa_fields = ['ts', 'aa_px', 'aa_py', 'aa_ppsi']
rh_fields = ['ts', 'rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']  # TODO -- what about rh velocity? not recorded during these experiments?

def preprocess_data(df):
    df = df.rename(field_mapping)

    df.ix['ts'] *= ms_to_us
    df.ix[['aa_px', 'aa_py']] *= mm_to_cm
    df.ix[['aa_ppsi', 'rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']] *= deg_to_rad
    
    starting_pos = np.array([21., 15.])
    pos_offset = df.ix[['aa_px', 'aa_py'], 0] - starting_pos

    df.ix['aa_px', :] -= pos_offset[0]
    df.ix['aa_py', :] -= pos_offset[1]

    return df


mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
kin_epoched = mat['kin_epoched']

traj = dict()
for i, kin in enumerate(kin_epoched):
    df = preprocess_data(pd.DataFrame(kin.T, index=index))

    if i % 2 == 0:
        trial_type = 'touch red'
    else:
        trial_type = 'pinch grip green'

    if not (trial_type in traj):
        traj[trial_type] = dict()
        
        ts_start = df.ix['ts', 0]
        ts_end   = df.ix['ts', df.columns[-1]]

        traj[trial_type]['ts_start']  = ts_start
        traj[trial_type]['ts_end']    = ts_end
        traj[trial_type]['armassist'] = df.ix[aa_fields, :]
        traj[trial_type]['rehand']    = df.ix[rh_fields, :]

pickle.dump(traj, open(pkl_name, 'wb'))
