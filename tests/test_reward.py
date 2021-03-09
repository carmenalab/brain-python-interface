"""
Modified by Pavi for Orsborn ab reward system
on Aug 2020 based on Si Jia's previous script

"""
from riglib import reward

import unittest

class TestAoReward(unittest.TestCase):

    def setUp(self):
        self.reward_sys = reward.Basic()

    def test_connection(self):
        self.assertTrue(self.reward_sys.board is not None)

    def test_flow_out(self):
        self.reward_sys.drain(0.2)

if __name__ == '__main__':
    unittest.main()


