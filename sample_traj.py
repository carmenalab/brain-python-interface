import scipy.io as sio
import numpy as np
import pandas as pd
import tables
import pickle

from utils.constants import *


pkl_name = 'traj_reference.pkl'
mat_name = '/home/lab/Desktop/Kinematic data ArmAssist/Epoched data/epoched_kin_data/NI_sess05_20140610/NI_B1S005R01.mat'


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

aa_xy_states   = ['aa_px', 'aa_py']
aa_pos_states  = ['aa_px', 'aa_py', 'aa_ppsi']
rh_pos_states  = ['rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']
rh_vel_states  = ['rh_vthumb', 'rh_vindex', 'rh_vfing3', 'rh_vprono']
ang_pos_states = ['aa_ppsi', 'rh_pthumb', 'rh_pindex', 'rh_pfing3', 'rh_pprono']

# ArmAssist and ReHand trajectories are saved as separate pandas DataFrames with the following indexes
aa_fields = ['ts'] + aa_pos_states
rh_fields = ['ts'] + rh_pos_states + rh_vel_states


def preprocess_data(df):
    # rename dataframe fields to match state space names used in Python code
    df = df.rename(field_mapping)

    # convert units to usec, cm, rad
    df.ix['ts'] *= ms_to_us
    df.ix[aa_xy_states] *= mm_to_cm
    df.ix[ang_pos_states] *= deg_to_rad
    
    # translate ArmAssist trajectories to start at a particular position
    starting_pos = np.array([21., 15., 0.])
    pos_offset = df.ix[aa_pos_states, 0] - starting_pos
    df.ix['aa_px', :]   -= pos_offset[0]
    df.ix['aa_py', :]   -= pos_offset[1]
    df.ix['aa_ppsi', :] -= pos_offset[2]

    # translate ReHand trajectories to start at a particular angular position
    starting_pos = deg_to_rad * np.array([30., 30., 30., 20.])
    pos_offset = df.ix[rh_pos_states, 0] - starting_pos
    df.ix['rh_pthumb', :] -= pos_offset[0]
    df.ix['rh_pindex', :] -= pos_offset[1]
    df.ix['rh_pfing3', :] -= pos_offset[2]
    df.ix['rh_pprono', :] -= pos_offset[3]

    # differentiate ReHand positions to get ReHand velocity data
    delta_pos = np.diff(df.ix[rh_pos_states, :])
    delta_ts  = us_to_s * np.diff(df.ix['ts', :])
    vel = np.hstack([np.zeros((4, 1)), delta_pos / delta_ts])
    df_rh_vel = pd.DataFrame(vel, index=rh_vel_states)
    df = pd.concat([df, df_rh_vel])

    return df


# load kinematic data
mat = sio.loadmat(mat_name, struct_as_record=False, squeeze_me=True)
kin_epoched = mat['kin_epoched']

# create a dictionary of trajectories, indexed by trial_type
traj = dict()
for i, kin in enumerate(kin_epoched):
    df = preprocess_data(pd.DataFrame(kin.T, index=index))

    # assign an arbitrary trial_type name
    if i % 2 == 0:
        trial_type = 'touch red'
    else:
        trial_type = 'pinch grip green'

    # if we haven't already saved a trajectory for this trial_type
    if trial_type not in traj:
        traj[trial_type] = dict()
        traj[trial_type]['ts_start']  = df.ix['ts', 0]
        traj[trial_type]['ts_end']    = df.ix['ts', df.columns[-1]]
        traj[trial_type]['armassist'] = df.ix[aa_fields, :]
        traj[trial_type]['rehand']    = df.ix[rh_fields, :]

pickle.dump(traj, open(pkl_name, 'wb'))
