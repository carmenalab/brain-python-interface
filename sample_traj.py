import scipy.io as sio
import numpy as np
import pandas as pd
import tables
import pickle
from scipy.interpolate import interp1d
import os

from riglib.ismore import settings
from utils.constants import *


pkl_name = 'traj_reference_interp.pkl'
mat_name = os.path.expandvars('$HOME/Desktop/Kinematic data ArmAssist/Epoched data/epoched_kin_data/NI_sess05_20140610/NI_B1S005R01.mat')


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
pos_states     = aa_pos_states + rh_pos_states


# ArmAssist and ReHand trajectories are saved as separate pandas DataFrames with the following indexes
aa_fields = ['ts'] + aa_pos_states
rh_fields = ['ts'] + rh_pos_states + rh_vel_states


def preprocess_data(df):
    # rename dataframe fields to match state space names used in Python code
    df = df.rename(field_mapping)

    # convert units to sec, cm, rad
    df.ix['ts'] *= ms_to_s
    df.ix[aa_xy_states] *= mm_to_cm
    df.ix[ang_pos_states] *= deg_to_rad
    
    # translate ArmAssist and ReHand trajectories to start at a particular position
    starting_pos = settings.starting_pos
    pos_offset = df.ix[pos_states, 0] - starting_pos
    for state in pos_states:
        df.ix[state, :] -= pos_offset[state]
    # TODO -- if df was time x states instead of states x time,
    #   then you could simply do: "df -= pos_offset"

    # differentiate ReHand positions to get ReHand velocity data
    delta_pos = np.diff(df.ix[rh_pos_states, :])
    delta_ts  = np.diff(df.ix['ts', :])
    vel = delta_pos / delta_ts
    vel = np.hstack([np.zeros((4, 1)), vel])
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

    ts_start = df.ix['ts', 0]
    ts_end   = df.ix['ts', df.columns[-1]]

    # assign an arbitrary trial_type name
    trial_type = 'touch %d' % (i % 4)

    # if we haven't already saved a trajectory for this trial_type
    if trial_type not in traj:
        traj[trial_type] = dict()
        traj[trial_type]['ts_start']  = ts_start
        traj[trial_type]['ts_end']    = ts_end

        # NEW: finely-spaced vector of time-stamps onto which we will interpolate armassist and rehand data
        ts_step = 5000  # microseconds
        ts_vec = np.arange(ts_start, ts_end, ts_step)

        df_final = pd.DataFrame(ts_vec, columns=['ts']).T
        
        for state in aa_pos_states + rh_pos_states + rh_vel_states:
            ts_data    = df.ix['ts', :]
            state_data = df.ix[state, :]
            interp_fn = interp1d(ts_data, state_data)
            df_tmp = pd.DataFrame(interp_fn(ts_vec), columns=[state]).T
            df_final = pd.concat([df_final, df_tmp])
        traj[trial_type]['traj'] = df_final

        # a bit repetitive -- also save aa and rh separately with their own ts
        df_aa = pd.DataFrame(ts_vec, columns=['ts']).T
        for state in aa_pos_states:
            ts_data    = df.ix['ts', :]
            state_data = df.ix[state, :]
            interp_fn = interp1d(ts_data, state_data)
            df_tmp = pd.DataFrame(interp_fn(ts_vec), columns=[state]).T
            df_aa  = pd.concat([df_aa, df_tmp])
        traj[trial_type]['armassist'] = df_aa
        
        df_rh = pd.DataFrame(ts_vec.T, columns=['ts']).T
        for state in rh_pos_states+rh_vel_states:
            ts_data    = df.ix['ts', :]
            state_data = df.ix[state, :]
            interp_fn = interp1d(ts_data, state_data)
            df_tmp = pd.DataFrame(interp_fn(ts_vec), columns=[state]).T
            df_rh  = pd.concat([df_rh, df_tmp])
        traj[trial_type]['rehand'] = df_rh
        

pickle.dump(traj, open(pkl_name, 'wb'))
