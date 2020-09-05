from riglib import ao_reward

import unittest

class TestAoReward(unittest.TestCase):

    def setUp(self):
        self.reward_sys = ao_reward.Basic()


    @unittest.skip("not sure which method to use")
    def test_connection(self):
        pass

    def test_flow_out(self):
        reward_time_ms  = 1000 #ms
        self.reward_sys(reward_time_ms)

    @unittest.skip("not sure how to calibrate yet")
    def test_calibration(self):
        print('fill up bottle')


if __name__ == '__main__':
    unittest.main()


