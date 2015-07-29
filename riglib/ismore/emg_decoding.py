import tables
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

from ismore.common_state_lists import *
from tasks import ismore_bmi_lib


# decodes velocity
class EMGDecoderBase(object):
    '''Abstract base class for all concrete EMG decoder classes.'''

    pass


class LinearEMGDecoder(EMGDecoderBase):
    '''Concrete base class for a linear EMG decoder.'''

    def __init__(self, channels, plant_type, fs, win_len, extractor_cls, extractor_kwargs):
        self.channels   = channels
        self.plant_type = plant_type
        self.fs         = fs
        self.win_len    = win_len

        # channel names in HDF file have 'chan' in front (e.g., 'chanBiceps'
        self.channel_names = ['chan' + name for name in channels]

        ssm = ismore_bmi_lib.SSM_CLS_DICT[plant_type]()
        self.states_to_decode = [s.name for s in ssm.states if s.order == 1]

        self.extractor_cls = extractor_cls
        self.extractor_kwargs = extractor_kwargs

    def __call__(self, features):
        decoder_output = pd.Series(0.0, self.states_to_decode)
        for state in self.states_to_decode:
            decoder_output[state] = self.beta[state].dot(features)

        return decoder_output

    def train_ridge(self, K, train_hdf_names, test_hdf_names, states_to_flip):
        '''Use ridge regression to train this decoder from data from multiple .hdf files.'''

        # save this info as part of the decoder object
        self.K               = K
        self.train_hdf_names = train_hdf_names
        self.test_hdf_names  = test_hdf_names
        self.states_to_flip  = states_to_flip

        # will be 2-D arrays, each with shape (N, n_features)
        # e.g., if extracting 7 features from each of 14 channels, then the
        #   shape might be (10000, 98)
        feature_data_train = None
        feature_data_test  = None

        # each will be a dictionary where:
        # key: a kinematic state (e.g., 'aa_px')
        # value: kinematic data for that state, interpolated to correspond to
        #   the same times as the rows of feature_data
        #   e.g., if feature_data_train has N rows, then each value in
        #   kin_data_train will be an array of length N
        kin_data_train = dict()
        kin_data_test  = dict()

        f_extractor = self.extractor_cls(None, **self.extractor_kwargs)

        all_hdf_names = train_hdf_names + [name for name in test_hdf_names if name not in train_hdf_names]

        for hdf_name in all_hdf_names:
            # load EMG data from HDF file
            hdf = tables.openFile(hdf_name)
            try:
                emg = hdf.root.brainamp[:][self.channel_names]
            except:  # in older HDF files, brainamp data was stored under table 'emg'
                emg = hdf.root.emg[:][self.channel_names]

            # "correct" the saved vector of timestamps by assuming that the
            #   last occurrence of the first EMG timestamp is correct
            #   e.g., if fs = 1000, EMG data arrives in blocks of 4 points, and
            #      the saved timestamps are:
            #        [5.103, 5.103, 5.103, 5.103, 5.107, 5.107, ...]
            #      then the "corrected" timestamps (saved in ts_vec) would be:
            #        [5.100, 5.101, 5.102, 5.103, 5.104, 5.105, ...]
            original_ts = emg[self.channel_names[0]]['ts_arrival']
            idx = 1
            while original_ts[idx] == original_ts[0]:
                idx = idx + 1
            idx = idx - 1
            ts_step = 1./self.fs
            ts_before = original_ts[idx] + (ts_step * np.arange(-idx, 0))
            ts_after = original_ts[idx] + (ts_step * np.arange(1, len(original_ts)))
            ts_vec = np.hstack([ts_before, original_ts[idx], ts_after])

            # cut off small amount of data from start and end of emg
            cutoff_time = 1  # secs
            cutoff_pts = int(cutoff_time * self.fs)
            emg = emg[cutoff_pts:-cutoff_pts]
            ts_vec = ts_vec[cutoff_pts:-cutoff_pts]

            n_win_pts = int(self.win_len * self.fs)
            step_pts = 100  # TODO -- don't hardcode
            start_idxs = np.arange(0, len(emg) - n_win_pts + 1, step_pts)

            features = np.zeros((len(start_idxs), f_extractor.n_features))
            for i, start_idx in enumerate(start_idxs):
                end_idx = start_idx + n_win_pts

                # samples has shape (n_chan, n_win_pts) 
                samples = np.vstack([emg[start_idx:end_idx][chan]['data'] for chan in self.channel_names])
                features[i, :] = f_extractor.extract_features(samples).T

            if hdf_name in train_hdf_names:
                if feature_data_train is None:
                    feature_data_train = features.copy()
                else:
                    feature_data_train = np.vstack([feature_data_train, features])
            if hdf_name in test_hdf_names:
                if feature_data_test is None:
                    feature_data_test = features.copy()
                else:
                    feature_data_test = np.vstack([feature_data_test, features])

            # we will interpolate ArmAssist and/or ReHand data at the times in ts_features
            ts_features = ts_vec[start_idxs + n_win_pts - 1]

            # TODO -- a lot of code is repeated below, find way to reduce

            if self.plant_type in ['ArmAssist', 'IsMore']:
                if 'armassist' not in hdf.root:
                    raise Exception('Invalid plant_type %s: no ArmAssist data saved in HDF file.' % self.plant_type)

                for (pos_state, vel_state) in zip(aa_pos_states, aa_vel_states):
                    # differentiate ArmAssist position data to get velocity; 
                    # the ArmAssist application doesn't send velocity 
                    #   feedback data, so it is not saved in the HDF file
                    delta_pos = np.diff(hdf.root.armassist[:]['data'][pos_state])
                    delta_ts  = np.diff(hdf.root.armassist[:]['ts'])
                    vel_state_data = delta_pos / delta_ts
                    ts_data = hdf.root.armassist[1:]['ts_arrival']
                    interp_fn = interp1d(ts_data, vel_state_data)
                    interp_state_data = interp_fn(ts_features)
                    if hdf_name in train_hdf_names:
                        try:
                            kin_data_train[vel_state] = np.concatenate([kin_data_train[vel_state], interp_state_data])
                        except KeyError:
                            kin_data_train[vel_state] = interp_state_data.copy()
                    if hdf_name in test_hdf_names:
                        try:
                            kin_data_test[vel_state] = np.concatenate([kin_data_test[vel_state], interp_state_data])
                        except KeyError:
                            kin_data_test[vel_state] = interp_state_data.copy()

            if self.plant_type in ['ReHand', 'IsMore']:
                if 'rehand' not in hdf.root:
                    raise Exception('Invalid plant_type %s: no ReHand data saved in HDF file.' % self.plant_type)

                for vel_state in rh_vel_states:
                    ts_data    = hdf.root.rehand[:]['ts_arrival']
                    state_data = hdf.root.rehand[:]['data'][vel_state]
                    interp_fn = interp1d(ts_data, state_data)
                    interp_state_data = interp_fn(ts_features)
                    if hdf_name in train_hdf_names:
                        try:
                            kin_data_train[vel_state] = np.concatenate([kin_data_train[vel_state], interp_state_data])
                        except KeyError:
                            kin_data_train[vel_state] = interp_state_data.copy()
                    if hdf_name in test_hdf_names:
                        try:
                            kin_data_test[vel_state] = np.concatenate([kin_data_test[vel_state], interp_state_data])
                        except KeyError:
                            kin_data_test[vel_state] = interp_state_data.copy()

        self.features_mean = np.mean(feature_data_train, axis=0)
        self.features_std = np.std(feature_data_train, axis=0)
        Z_features_train = (feature_data_train - self.features_mean) / self.features_std

        # train vector of coefficients for each DoF using ridge regression
        self.beta = dict()
        for state in kin_data_train:
            self.beta[state] = ridge(kin_data_train[state], Z_features_train, K, zscore=False)

        # test coefficients for each DoF on testing data
        Z_features_test = (feature_data_test - self.features_mean) / self.features_std
        cc_values = dict()
        for state in kin_data_test:
            pred_kin_data = np.dot(Z_features_test, self.beta[state])
            cc_values[state] = pearsonr(kin_data_test[state], pred_kin_data)[0]

        print cc_values

        # TODO -- set lambda_coeffs manually for now
        self.lambda_coeffs = pd.Series(0.0, self.states_to_decode)
        if self.plant_type in ['ArmAssist', 'IsMore']:
            self.lambda_coeffs['aa_vx']     = 0.9
            self.lambda_coeffs['aa_vy']     = 0.9
            self.lambda_coeffs['aa_vpsi']   = 0.9
        if self.plant_type in ['ReHand', 'IsMore']:
            self.lambda_coeffs['rh_vthumb'] = 0.9
            self.lambda_coeffs['rh_vindex'] = 0.9
            self.lambda_coeffs['rh_vfing3'] = 0.9
            self.lambda_coeffs['rh_vprono'] = 0.9

        for state in states_to_flip:
            if state in self.beta:
                self.beta[state] *= -1.0


def ridge(Y, X, K, zscore=True):
    '''Same as MATLAB's ridge function.'''

    p = X.shape[1]
    if zscore:
        Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    else:
        Z = X
    return np.linalg.pinv(Z.T.dot(Z) + K*np.identity(p)).dot(Z.T).dot(Y)



