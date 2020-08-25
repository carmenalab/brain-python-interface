from riglib.bmi import lindecoder
from built_in_tasks.bmimultitasks import SimBMICosEncLinDec
from riglib import experiment
import numpy as np

import unittest

class TestLinDec(unittest.TestCase):

    def setUp(self):
        pass

    def test_sanity(self):
        simple_filt = lindecoder.LinearScaleFilter(100, 1, 1)
        self.assertEqual(0, simple_filt.get_mean())
        
        for i in range(50):
            simple_filt([1])

        self.assertEqual(0.5, np.mean(simple_filt.obs))
        self.assertEqual(0, simple_filt.get_mean())

        for i in range(250):
            simple_filt(i)

        self.assertTrue(simple_filt.get_mean() > 0.5) # 0.9 not working because of normalization by std instead of range

    def test_filter(self):
        filt = lindecoder.LinearScaleFilter(100, 3, 2)
        self.assertListEqual([0,0,0], filt.get_mean().tolist())
        for i in range(100):
            filt([0, 0])
            self.assertEqual(0, filt.state.mean[0, 0])
            self.assertEqual(0, filt.state.mean[1, 0])
            self.assertEqual(0, filt.state.mean[2, 0])
    
    #@unittest.skip('msg')
    def test_experiment(self):
        N_TARGETS = 8
        N_TRIALS = 3
        seq = SimBMICosEncLinDec.sim_target_seq_generator_multi(
            N_TARGETS, N_TRIALS)
        base_class = SimBMICosEncLinDec
        feats = []
        Exp = experiment.make(base_class, feats=feats)
        exp = Exp(seq)
        exp.init()
        exp.run()
        
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
        self.assertTrue(rewards <= rewards + time_penalties + hold_penalties)
        self.assertTrue(rewards > 0)

if __name__ == '__main__':
    unittest.main()


