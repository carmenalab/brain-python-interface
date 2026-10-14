from analysis import online_analysis
from features.clda_features import CLDA_KFRML_IntendedVelocity
from features.debug_features import OnlineAnalysis
from riglib.bmi import lindecoder, kfdecoder, wfdecoder, state_space_models, extractor, train
from built_in_tasks.bmimultitasks import BMIControlMulti #, SimBMICosEncLinDec, SimBMIVelocityLinDec
from features.ecube_features import EcubeBMI, EcubeFileBMI, RecordECube
from features.neural_sys_features import SpikerBoxBMI
from riglib.spikerbox import LFP
from riglib.stereo_opengl.window import Window2D, WindowDispl2D
from riglib import experiment
from features.hdf_features import SaveHDF
import numpy as np
import sys

import unittest


# might need this on windows:
# export LIBGL_ALWAYS_INDIRECT=0
#export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0

class TestKFDecoder(unittest.TestCase):
        
    @unittest.skip("")
    def test_fixed_decoder_ecube(self):

         # Construct a fixed decoder
        ssm = state_space_models.StateSpaceEndptVel2D()
        units = np.array([[1, 0], [2, 0]])
        C = np.zeros([2, 7])
        C[0, 3] = 0.1
        C[1, 5] = 0.1
        decoder = train.make_fixed_kf_decoder(units, ssm, C, dt=0.1)
        decoder.extractor_cls = extractor.LFPMTMPowerExtractor
        decoder.extractor_kwargs = dict(channels=[1, 2], bands=[(90,110)], win_len=0.1, fs=1000)

        import pickle
        import os
        test_decoder_filename = os.path.join('tests', 'test_kf_decoder.pkl')
        with open(test_decoder_filename, 'wb') as f:
            pickle.dump(decoder, f, 2)

        del decoder
        
        with open(test_decoder_filename, 'rb') as f:
            decoder = pickle.load(f)

        base_class = BMIControlMulti

        # Settings for streaming from ecube
        # feats = [EcubeBMI, WindowDispl2D, SaveHDF] # use default headstage port 7
        # kwargs = dict(decoder=decoder, window_size=(500,500), fullscreen=False)

        # Settings for reading from file
        feats = [EcubeFileBMI, Window2D, SaveHDF]
        test_file = 'tests/test_data/simple'
        # test_file = '/media/server/raw/ecube/ecube test data'
        kwargs = dict(ecube_bmi_filename=test_file, decoder=decoder)

        seq = base_class.centerout_2D(nblocks=1, ntargets=8, distance=8)
        Exp = experiment.make(base_class, feats=feats)
        exp = Exp(seq, **kwargs)

        exp.window_size = (500,500)
        exp.fullscreen = False
        exp.init()

        print(f"decoder units: {exp.decoder.units}")
        print(f"decoder binlen: {exp.decoder.binlen}")
        print(f"decoder call rate: {exp.decoder.call_rate}")

        exp.run()

        h5file = exp.get_h5_filename()
        os.rename(h5file, 'test_decoder.hdf')
        
        rewards, time_penalties, hold_penalties = calculate_rewards(exp)
        self.assertTrue(rewards <= rewards + time_penalties + hold_penalties)
        self.assertTrue(rewards >= 0)
   
    def test_fixed_decoder_spikerbox(self):

         # Construct a fixed decoder
        ssm = state_space_models.StateSpaceEndptVel2D()
        units = np.array([[1, 0], [2, 0]])
        C = np.zeros([2, 7])
        C[0, 3] = 0.1
        C[1, 5] = 0.1
        decoder = train.make_fixed_kf_decoder(units, ssm, C, dt=0.1)
        decoder.extractor_cls = extractor.LFPMTMPowerExtractor
        decoder.extractor_kwargs = dict(channels=[1, 2], bands=[(50,500)], win_len=0.1, fs=10000, ref=False)
        decoder.init_zscore(np.array([2.5, 1.7]), np.array([1., 1.]))

        import pickle
        import os
        test_decoder_filename = os.path.join('tests', 'test_emg_decoder.pkl')
        with open(test_decoder_filename, 'wb') as f:
            pickle.dump(decoder, f, 2)

        del decoder
        with open(test_decoder_filename, 'rb') as f:
            decoder = pickle.load(f)

        base_class = BMIControlMulti
        analysis = online_analysis.OnlineDataServer('localhost', 5000)
        analysis.start()

        feats = [SpikerBoxBMI, CLDA_KFRML_IntendedVelocity, Window2D, SaveHDF, OnlineAnalysis]
        kwargs = dict(decoder=decoder, clda_batch_time=1, clda_update_half_life=5)

        seq = base_class.centerout_2D(nblocks=1, ntargets=2, distance=8)
        Exp = experiment.make(base_class, feats=feats)
        exp = Exp(seq, **kwargs)

        exp.window_size = (800,500)
        exp.fullscreen = False
        exp.init()

        print(f"decoder units: {exp.decoder.units}")
        print(f"decoder binlen: {exp.decoder.binlen}")
        print(f"decoder call rate: {exp.decoder.call_rate}")
        exp.enable_clda()

        exp.run()

        h5file = exp.get_h5_filename()
        os.rename(h5file, 'test_decoder.hdf')
        
        # Wrap up
        analysis.stop()
        analysis.join()

        rewards, time_penalties, hold_penalties = calculate_rewards(exp)
        self.assertTrue(rewards <= rewards + time_penalties + hold_penalties)
        self.assertTrue(rewards >= 0)

    @unittest.skip('msg')
    def test_trained_decoder_simulation(self):
        import aopy

        # Train a decoder form test neural and cursor data generated from a known encoder model
        data = aopy.data.load_hdf_group('tests/test_data/feature_selection', 'wo_FS_0.7_training_data.hdf')
        position = data['kinematics']
        velocity = np.diff(position.T, axis=0) * 1./(1/60)
        velocity = np.vstack([np.zeros(position.shape[0]), velocity])
        kin = np.hstack([position.T, velocity])
        print(kin.shape)
        units = np.array([[i+1, 0] for i in range(8)])
        neural_features = data['spike_counts']
        print(neural_features.shape)
        update_rate = 60
        ssm = state_space_models.StateSpaceEndptVel2D()
        decoder = train.train_KFDecoder_abstract(ssm, kin.T, neural_features.T, units, update_rate)

        print(np.round(decoder.kf.C, 3))
        print(np.round(decoder.kf.Q, 3))

    @unittest.skip('msg')
    def test_trained_decoder_ecube(self):
        import aopy

        # Train a decoder form test neural and cursor data generated from a known encoder model
        data = aopy.data.load_hdf_group('tests/test_data/feature_selection', 'wo_FS_0.7_training_data.hdf')
        winlen = 0.5
        position = data['kinematics'][:,::int(winlen*60)] # only need one kin sample per extractor window size
        velocity = np.diff(position.T, axis=0) * 1./(1/60)
        velocity = np.vstack([np.zeros(position.shape[0]), velocity])
        kin = np.hstack([position.T, velocity])
        print(kin.shape)
        files = {'ecube': 'tests/test_data/feature_selection'}
        units = np.array([[i+1, 0] for i in range(8)])
        print(data['neurows'].shape)
        print(data['neurows'])
        neural_features, units, extractor_kwargs = extractor.LFPMTMPowerExtractor.extract_from_file(files, data['neurows'], winlen, units, {'channels': [i+1 for i in range(8)], 'bands': [(70,90)], 'win_len': winlen})
        print(neural_features.shape)
        
        import matplotlib.pyplot as plt
        plt.plot(neural_features[:,2])
        plt.plot(kin[:,2])
        plt.show()

        update_rate = 60
        ssm = state_space_models.StateSpaceEndptVel2D()
        decoder = train.train_KFDecoder_abstract(ssm, kin.T, neural_features.T, units, update_rate, zscore=True)
        decoder.extractor_cls = extractor.LFPMTMPowerExtractor
        decoder.extractor_kwargs = extractor_kwargs
        print(np.round(decoder.kf.C, 3))
        print(np.round(decoder.kf.Q, 3))

        # Save the decoder
        import pickle
        import os
        test_decoder_filename = os.path.join('tests', 'trained_kf_decoder.pkl')
        with open(test_decoder_filename, 'wb') as f:
            pickle.dump(decoder, f, 2)

        # Load the sequence
        data = aopy.data.load_hdf_group('tests/test_data/feature_selection', 'wo_FS_0.7_training_data.hdf')
        targ_seq = data['target_sequence']
        targ_locs = data['target_location']
        seq = list(zip([[i] for i in targ_seq], [[l] for l in targ_locs]))

        # Make and run the experiment
        base_class = BMIControlMulti
        #feats = [EcubeBMI] # use default headstage port 7
        feats = [EcubeFileBMI, Window2D]
        kwargs = dict(ecube_bmi_filename='tests/test_data/feature_selection', decoder=decoder)
        Exp = experiment.make(base_class, feats=feats)
        exp = Exp(seq, **kwargs)

        exp.window_size = (500,500)
        exp.fullscreen = False

        print(exp.decoder.units)
        print(exp.decoder.units[:,0])
        print(exp.cortical_channels)

        exp.init()
        exp.run()

        # Do some simple checks
        rewards, time_penalties, hold_penalties = calculate_rewards(exp)
        self.assertTrue(rewards <= rewards + time_penalties + hold_penalties)
        self.assertTrue(rewards > 0)


class TestLinDec(unittest.TestCase):

    def test_sanity(self):
        simple_filt = lindecoder.LinearScaleFilter(100, 1, 1)
        self.assertEqual(0, simple_filt.get_mean())
        
        for i in range(50):
            simple_filt([1])

        self.assertEqual(0.5, np.mean(simple_filt.obs))
        self.assertEqual(0, simple_filt.get_mean())

        for i in range(250):
            simple_filt(i)

        self.assertTrue(simple_filt.get_mean() > 0)

    def test_filter(self):
        filt = lindecoder.LinearScaleFilter(100, 3, 2)
        self.assertListEqual([0,0,0], filt.get_mean().tolist())
        for i in range(100):
            filt([0, 0])
            self.assertEqual(0, filt.state.mean[0, 0])
            self.assertEqual(0, filt.state.mean[1, 0])
            self.assertEqual(0, filt.state.mean[2, 0])
    
    @unittest.skip('msg')
    def test_experiment_unfixed(self):
        from built_in_tasks.bmimultitasks import SimBMICosEncLinDec, SimBMIVelocityLinDec
        for cls in [SimBMICosEncLinDec]:
            N_TARGETS = 8
            N_TRIALS = 16
            seq = cls.sim_target_no_center(
                N_TARGETS, N_TRIALS)
            base_class = cls
            feats = []
            Exp = experiment.make(base_class, feats=feats)
            exp = Exp(seq)
            exp.init()

            exp.run()
            
            rewards, time_penalties, hold_penalties = calculate_rewards(exp)
            self.assertTrue(rewards <= rewards + time_penalties + hold_penalties)
            self.assertTrue(rewards > 0)
    
    @unittest.skip('msg')
    def test_experiment(self):
        from built_in_tasks.bmimultitasks import SimBMICosEncLinDec, SimBMIVelocityLinDec
        for cls in [SimBMICosEncLinDec, SimBMIVelocityLinDec]:
            N_TARGETS = 8
            N_TRIALS = 16
            seq = cls.sim_target_seq_generator_multi(
                N_TARGETS, N_TRIALS)
            base_class = cls
            feats = []
            Exp = experiment.make(base_class, feats=feats)
            exp = Exp(seq)
            exp.init()
            exp.decoder.filt.fix_norm_attr()

            exp.run()
            
            rewards, time_penalties, hold_penalties = calculate_rewards(exp)
            self.assertTrue(rewards <= rewards + time_penalties + hold_penalties)
            self.assertTrue(rewards > 0)



class TestWFDecoder(unittest.TestCase):

    def setUp(self):
        '''
        Simulate a population whose activity is linearly related to the velocity
        in the current bin and in the previous bin
        '''
        np.random.seed(0)
        self.ssm = state_space_models.StateSpaceEndptVel2D()
        self.n_units = 12
        self.T = 2000
        self.units = np.vstack([np.arange(1, self.n_units+1), np.zeros(self.n_units)]).T.astype(np.int32)

        self.neural_features = np.random.randn(self.n_units, self.T) + 5
        B_0 = np.random.randn(2, self.n_units)
        B_1 = np.random.randn(2, self.n_units)
        vel = B_0.dot(self.neural_features) + \
            np.hstack([np.zeros((2, 1)), B_1.dot(self.neural_features[:, :-1])])

        self.kin = np.zeros((6, self.T))
        self.kin[3, :] = vel[0, :]
        self.kin[5, :] = vel[1, :]

    def _train(self, **kwargs):
        return train.train_WFDecoder_abstract(self.ssm, self.kin, self.neural_features,
            self.units, 0.1, **kwargs)

    def test_obs_history(self):
        obs = np.arange(6).reshape(2, 3)
        obs_hist = wfdecoder.WienerFilter.form_obs_history(obs, n_taps=2)

        # each column is [y_t; y_{t-1}; 1], with the first column zero-padded
        np.testing.assert_array_equal(obs_hist[:, 0], [0, 3, 0, 0, 1])
        np.testing.assert_array_equal(obs_hist[:, 1], [1, 4, 0, 3, 1])
        np.testing.assert_array_equal(obs_hist[:, 2], [2, 5, 1, 4, 1])

    def test_one_tap_is_linear_regression(self):
        decoder = self._train(n_taps=1)

        obs = np.vstack([self.neural_features, np.ones(self.T)])
        H = np.linalg.lstsq(obs.T, self.kin[self.ssm.train_inds, :].T, rcond=None)[0].T
        np.testing.assert_allclose(np.asarray(decoder.filt.H)[self.ssm.train_inds, :], H)

        # states which aren't estimated from the observations have no filter weights
        self.assertTrue(np.all(np.asarray(decoder.filt.H)[[0, 1, 2, 4, 6], :] == 0))

    def test_decoding(self):
        decoder = self._train(n_taps=4)
        self.assertEqual(decoder.filt.H.shape, (self.ssm.n_states, self.n_units*4 + 1))
        self.assertEqual(decoder.filt.n_features, self.n_units)

        decoder.filt._init_state()
        out = decoder.decode(self.neural_features)

        # the simulated velocity is exactly a linear function of the neural history
        for state, kin_ind in zip([3, 5], [3, 5]):
            corr = np.corrcoef(out[4:, state], self.kin[kin_ind, 4:])[0, 1]
            self.assertTrue(corr > 0.99)

        # one tap can't capture the lagged component of the simulated tuning
        decoder_1tap = self._train(n_taps=1)
        decoder_1tap.filt._init_state()
        out_1tap = decoder_1tap.decode(self.neural_features)
        self.assertTrue(np.corrcoef(out_1tap[:, 3], self.kin[3, :])[0, 1] <
                        np.corrcoef(out[4:, 3], self.kin[3, 4:])[0, 1])

    def test_state_space_update(self):
        decoder = self._train(n_taps=4)
        decoder.filt._init_state()
        out = decoder.decode(self.neural_features)

        # states which aren't estimated from the observations follow the state space model
        np.testing.assert_allclose(out[:, 6], 1)
        np.testing.assert_allclose(out[:, 4], 0)
        np.testing.assert_allclose(out[1:, 0], np.cumsum(out[:-1, 3])*decoder.binlen)

    def test_regularization(self):
        decoder = self._train(n_taps=4)
        decoder_ridge = self._train(n_taps=4, regularizer=1e4)

        # the offset term (the last column of H) is not penalized
        self.assertTrue(np.linalg.norm(decoder_ridge.filt.H[:, :-1]) <
                        np.linalg.norm(decoder.filt.H[:, :-1]))

    def test_pickle(self):
        import pickle
        decoder = self._train(n_taps=4)
        decoder.filt._init_state()
        out = decoder.decode(self.neural_features)

        decoder_copy = pickle.loads(pickle.dumps(decoder, 2))
        np.testing.assert_array_equal(decoder_copy.filt.H, decoder.filt.H)
        self.assertEqual(decoder_copy.filt.n_taps, decoder.filt.n_taps)
        np.testing.assert_array_equal(decoder_copy.filt.is_stochastic, decoder.filt.is_stochastic)

        decoder_copy.filt._init_state()
        np.testing.assert_allclose(out, decoder_copy.decode(self.neural_features))

    def test_call(self):
        decoder = self._train(n_taps=4)
        decoder.filt._init_state()

        # the task calls the decoder one observation at a time
        states = [decoder(self.neural_features[:, k].reshape(-1, 1)) for k in range(10)]
        self.assertEqual(states[0].shape, (self.ssm.n_states, 1))

        decoder.filt._init_state()
        np.testing.assert_allclose(np.hstack(states).T, decoder.decode(self.neural_features[:, :10]))

    def test_control_input(self):
        decoder = self._train(n_taps=4)
        obs = self.neural_features[:, 0].reshape(-1, 1)

        decoder.filt._init_state()
        state = np.array(decoder(obs)).ravel()

        # an assistive control input is added on to the states the filter estimates
        decoder.filt._init_state()
        Bu = np.asmatrix(np.zeros((self.ssm.n_states, 1)))
        Bu[3, 0] = 10.
        state_assist = np.array(decoder(obs, Bu=Bu)).ravel()
        np.testing.assert_allclose(state_assist[3], state[3] + 10.)
        np.testing.assert_allclose(state_assist[5], state[5])

    def test_zscore(self):
        decoder = self._train(n_taps=4, zscore=True)
        self.assertTrue(decoder.zscore)

        decoder.filt._init_state()
        out = decoder.decode(self.neural_features)
        self.assertTrue(np.corrcoef(out[4:, 3], self.kin[3, 4:])[0, 1] > 0.99)

    def test_fixed_decoder(self):
        n_taps = 2
        # only the rows of the trained states, without the offset column
        H = np.random.randn(2, self.n_units*n_taps)
        decoder = train.make_fixed_wf_decoder(self.units, self.ssm, H, dt=0.1, n_taps=n_taps)
        self.assertEqual(decoder.filt.H.shape, (self.ssm.n_states, self.n_units*n_taps + 1))
        self.assertEqual(decoder.filt.n_taps, n_taps)
        self.assertEqual(decoder.n_features, self.n_units)
        np.testing.assert_array_equal(np.asarray(decoder.filt.H)[[3, 5], :-1], H)
        self.assertTrue(np.all(np.asarray(decoder.filt.H)[[0, 1, 2, 4, 6], :] == 0))
        self.assertTrue(np.all(np.asarray(decoder.filt.H)[:, -1] == 0))

        # the full matrix is accepted as is
        decoder_full = train.make_fixed_wf_decoder(self.units, self.ssm, decoder.filt.H, dt=0.1, n_taps=n_taps)
        np.testing.assert_array_equal(decoder_full.filt.H, decoder.filt.H)

        with self.assertRaises(AssertionError):
            train.make_fixed_wf_decoder(self.units, self.ssm, H, dt=0.1, n_taps=3)

        # the decoder runs and behaves as a trained one
        decoder.filt._init_state()
        out = decoder.decode(self.neural_features[:, :20])
        self.assertEqual(out.shape, (20, self.ssm.n_states))
        np.testing.assert_allclose(out[1:, 0], np.cumsum(out[:-1, 3])*decoder.binlen)

        import pickle
        decoder_copy = pickle.loads(pickle.dumps(decoder, 2))
        np.testing.assert_array_equal(decoder_copy.filt.H, decoder.filt.H)

    def test_random_decoder(self):
        np.random.seed(1)
        decoder = train.rand_WFDecoder(self.ssm, self.units, dt=0.1, n_taps=3, scale=0.5)
        H = np.asarray(decoder.filt.H)
        self.assertEqual(H.shape, (self.ssm.n_states, self.n_units*3 + 1))
        self.assertTrue(np.all(H[[0, 1, 2, 4, 6], :] == 0))
        self.assertTrue(np.all(H[:, -1] == 0))
        self.assertAlmostEqual(np.std(H[[3, 5], :-1]), 0.5, delta=0.1)

        decoder.filt._init_state()
        out = decoder.decode(self.neural_features[:, :10])
        self.assertEqual(out.shape, (10, self.ssm.n_states))


def calculate_rewards(exp):
    rewards = 0
    time_penalties = 0
    hold_penalties = 0
    for s in exp.event_log:
        if s[0] == 'reward':
            rewards += 1
        elif s[0] == 'hold_penalty':
            hold_penalties += 1
        elif s[0] == 'timeout_penalty':
            time_penalties += 1
    return rewards, time_penalties, hold_penalties


if __name__ == '__main__':
    unittest.main()


