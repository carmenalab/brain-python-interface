"""
Modified by Pavi for Orsborn ab reward system
on Aug 2020 based on Si Jia's previous script

"""
from riglib import reward

import unittest

class TestAoReward(unittest.TestCase):

    def setUp(self):
        self.reward_sys = reward.Basic()

    #@unittest.skip("not sure which method to use")
    def test_connection(self):
        self.reward_sys.test()
        pass

    def test_flow_out(self):
        reward_time_s  = 0.2 #s
        self.reward_sys.reward(reward_time_s)

    #@unittest.skip("not sure how to calibrate yet")
    def test_calibration(self):
        self.reward_sys.calibrate(self)
        print('fill up bottle')


if __name__ == '__main__':
    unittest.main()


