"""
Modified by Pavi for Orsborn ab reward system
on Aug 2020 based on Si Jia's previous script

"""
from riglib import reward

import unittest

# class TestAoReward(unittest.TestCase):

#     def setUp(self):
#         self.reward_sys = reward.Basic()

#     def test_connection(self):
#         self.assertTrue(self.reward_sys.board is not None)

#     def test_calibrate(self):
#         import numpy as np
#         import time

#         n_reps = 10
#         for t in np.arange(0.1, 1.1, 0.1):
#             print("Testing " + str(t) + " seconds")
#             for i in range(n_reps):
#                 print(".", end="")
#                 self.reward_sys.drain(t)
#                 time.sleep(0.5)


class TestAoPellet(unittest.TestCase):

    def setUp(self):
        self.reward_sys = reward.pellet_open()

    def test_connection(self):
        self.assertTrue(self.reward_sys is not None)
        self.assertTrue(self.reward_sys.board is not None)

    def test_calibrate(self):
        import numpy as np
        import time

        n_reps = 10
        for i in range (n_reps):
            print(".", end="")
            self.reward_sys.dispense()
            time.sleep(1)


if __name__ == '__main__':
    unittest.main()



