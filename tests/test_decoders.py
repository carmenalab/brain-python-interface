from riglib.bmi import lindecoder, kfdecoder, state_space_models, extractor, train
from built_in_tasks.bmimultitasks import BMIControlMulti #, SimBMICosEncLinDec, SimBMIVelocityLinDec
from features.ecube_features import EcubeBMI, EcubeFileBMI
from riglib.stereo_opengl.window import WindowDispl2D
from riglib import experiment
import numpy as np

import unittest

def make_fixed_kf_decoder(units, ssm, C, dt=0.1):
    n_neurons = units.shape[0]
    assert n_neurons == C.shape[0], "C matrix must have same first dimension as number of neurons"
    binlen = dt

    A, B, W = ssm.get_ssm_matrices(update_rate=dt)
    drives_neurons = ssm.drives_obs
    is_stochastic = ssm.is_stochastic
    nX = ssm.n_states
    assert nX == C.shape[1], "C matrix must have same second dimension as number of states"

    Q = 10 * np.identity(n_neurons)

    kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=is_stochastic)

    mFR = 0
    sdFR = 1
    decoder = kfdecoder.KFDecoder(kf, units, ssm, mFR=mFR, sdFR=sdFR, binlen=binlen)

    decoder.kf.R = np.mat(np.identity(decoder.kf.C.shape[1]))
    decoder.kf.S = decoder.kf.C
    decoder.kf.T = decoder.kf.Q + decoder.kf.S*decoder.kf.S.T
    decoder.kf.ESS = 3000.

    decoder.ssm = ssm
    decoder.n_features = n_neurons

    # decoder.bounder = make_rect_bounder_from_ssm(ssm)

    return decoder


class TestKFDecoder(unittest.TestCase):
        
    def test_fixed_decoder_ecube(self):

         # Construct a fixed decoder
        ssm = state_space_models.StateSpaceEndptVel2D()
        units = np.array([[1, 0], [2, 0]])
        C = np.zeros([2, 7])
        C[0, 3] = 1.
        C[1, 5] = 1.
        decoder = make_fixed_kf_decoder(units, ssm, C, dt=0.1)
        decoder.extractor_cls = extractor.LFPMTMPowerExtractor
        decoder.extractor_kwargs = dict(channels=[1, 2], bands=[(50,80)], win_len=0.1, fs=1000)

        import pickle
        import os
        test_decoder_filename = os.path.join('tests', 'test_kf_decoder.pkl')
        with open(test_decoder_filename, 'wb') as f:
            pickle.dump(decoder, f, 2)

        base_class = BMIControlMulti
        #feats = [EcubeBMI] # use default headstage port 7
        feats = [EcubeFileBMI, WindowDispl2D]
        kwargs = dict(ecube_bmi_filename='tests/test_data/simple', decoder=decoder)
        seq = base_class.centerout_2D(nblocks=1, ntargets=8, distance=8)
        Exp = experiment.make(base_class, feats=feats)
        exp = Exp(seq, **kwargs)

        print(exp.decoder.units)
        print(exp.decoder.units[:,0])
        print(exp.cortical_channels)

        exp.init()

        exp.run()
        
        rewards, time_penalties, hold_penalties = calculate_rewards(exp)
        self.assertTrue(rewards <= rewards + time_penalties + hold_penalties)
        self.assertTrue(rewards > 0)

    @unittest.skip('msg')
    def test_trained_decoder_ecube(self):
        import aopy

        # Train a decoder form test neural and cursor data generated from a known encoder model
        data = aopy.data.load_hdf_group('tests/test_data', 'wo_FS_0.7_training_data.hdf')
        position = data['kinematics'][:,::int(0.1*60)] # only need one kin sample per extractor window size
        velocity = np.diff(position.T, axis=0) * 1./(1/60)
        velocity = np.vstack([np.zeros(position.shape[0]), velocity])
        kin = np.hstack([position.T, velocity])
        print(kin.shape)
        files = {'ecube': 'tests/test_data/feature_selection'}
        units = np.array([[i+1, 0] for i in range(8)])
        print(data['neurows'].shape)
        neural_features, units, extractor_kwargs = extractor.LFPMTMPowerExtractor.extract_from_file(files, data['neurows'], 0.1, units, {'channels': [i+1 for i in range(8)], 'bands': [(70,90)], 'win_len': 0.1})
        print(neural_features.shape)
        update_rate = 60
        decoder = train.train_KFDecoder_abstract(ssm, kin.T, neural_features.T, units, update_rate)
        decoder.extractor_cls = extractor.LFPMTMPowerExtractor
        decoder.extractor_kwargs = extractor_kwargs

        test_decoder_filename = os.path.join('tests', 'trained_kf_decoder.pkl')
        import os
        import pickle
        with open(test_decoder_filename, 'wb') as f:
            pickle.dump(decoder, f, 2)


        data = aopy.data.load_hdf_group('tests/test_data', 'wo_FS_0.7_training_data.hdf')
        targ_seq = data['target_sequence']
        targ_locs = data['target_location']
        seq = list(zip([[i] for i in targ_seq], [[l] for l in targ_locs]))

        base_class = BMIControlMulti
        #feats = [EcubeBMI] # use default headstage port 7
        feats = [EcubeFileBMI, WindowDispl2D]
        kwargs = dict(ecube_bmi_filename='tests/test_data/feature_selection', decoder=decoder)
        Exp = experiment.make(base_class, feats=feats)
        exp = Exp(seq, **kwargs)

        exp.window_size = (1920,1080)

        print(exp.decoder.units)
        print(exp.decoder.units[:,0])
        print(exp.cortical_channels)

        exp.init()
        exp.run()
        
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


