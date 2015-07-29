import scipy.io as sio
import numpy as np
import pandas as pd
import tables
import pickle
from scipy.interpolate import interp1d
import os

from ismore import settings
from utils.constants import *


pkl_name = os.path.expandvars('$BMI3D/riglib/ismore/traj_reference_interp.pkl')
mat_name = os.path.expandvars('$HOME/Desktop/Kinematic data ArmAssist/Epoched data/epoched_kin_data/NI_sess05_20140610/NI_B1S005R01.mat')


columns = [
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
pos_states     = aa_pos_states + rh_pos_states


# ArmAssist and ReHand trajectories are saved as separate pandas DataFrames with the following indexes
aa_fields = ['ts'] + aa_pos_states
rh_fields = ['ts'] + rh_pos_states + rh_vel_states


def preprocess_data(df):
    # rename dataframe fields to match state space names used in Python code
    df = df.rename(columns=field_mapping)

    # convert units to sec, cm, rad
    df['ts'] *= ms_to_s
    df[aa_xy_states] *= mm_to_cm
    df[ang_pos_states] *= deg_to_rad
    
    # translate ArmAssist and ReHand trajectories to start at a particular position
    starting_pos = settings.starting_pos
    pos_offset = df.ix[0, pos_states] - starting_pos
    df[pos_states] -= pos_offset

    # differentiate ReHand positions to get ReHand velocity data
    delta_pos = np.diff(df[rh_pos_states], axis=0)
    delta_ts  = np.diff(df['ts']).reshape(-1, 1)
    vel = delta_pos / delta_ts
    vel = np.vstack([np.zeros((1, 4)), vel])
    df_rh_vel = pd.DataFrame(vel, columns=rh_vel_states)
    df = pd.concat([df, df_rh_vel], axis=1)

    return df


# load kinematic data
mat = sio.loadmat(mat_name, struct_as_record=False, squeeze_me=True)
kin_epoched = mat['kin_epoched']

trial_types = ['Blue', 'Brown', 'Green', 'Red']

# create a dictionary of trajectories, indexed by trial_type
traj = dict()
for i, kin in enumerate(kin_epoched):
    df = pd.DataFrame(kin, columns=columns)
    df = preprocess_data(df)

    ts_start = df['ts'][0]
    ts_end   = df['ts'][df.index[-1]]

    # assign an arbitrary trial_type name
    trial_type = trial_types[i % len(trial_types)]

    # if we haven't already saved a trajectory for this trial_type
    if trial_type not in traj:
        traj[trial_type] = dict()
        traj[trial_type]['ts_start'] = ts_start
        traj[trial_type]['ts_end']   = ts_end

        # finely-spaced vector of time-stamps onto which we will interpolate armassist and rehand data
        ts_step = 5e-3  # seconds
        ts_interp = np.arange(ts_start, ts_end, ts_step)
        df_ts_interp = pd.DataFrame(ts_interp, columns=['ts'])

        # comment
        df_aa = df_ts_interp.copy()
        for state in aa_pos_states:
            interp_fn = interp1d(df['ts'], df[state])
            interp_state_data = interp_fn(ts_interp)
            df_tmp = pd.DataFrame(interp_state_data, columns=[state])
            df_aa  = pd.concat([df_aa, df_tmp], axis=1)

        traj[trial_type]['armassist'] = df_aa
        
        # comment
        df_rh = df_ts_interp.copy()
        for state in rh_pos_states + rh_vel_states:
            interp_fn = interp1d(df['ts'], df[state])
            interp_state_data = interp_fn(ts_interp)
            df_tmp = pd.DataFrame(interp_state_data, columns=[state])
            df_rh  = pd.concat([df_rh, df_tmp], axis=1)

        traj[trial_type]['rehand'] = df_rh

        # comment
        df_traj = df_ts_interp.copy()
        for state in aa_pos_states:
            df_traj = pd.concat([df_traj, df_aa[state]], axis=1)
        for state in rh_pos_states + rh_vel_states:
            df_traj = pd.concat([df_traj, df_rh[state]], axis=1)
        
        traj[trial_type]['traj'] = df_traj
        

pickle.dump(traj, open(pkl_name, 'wb'))
