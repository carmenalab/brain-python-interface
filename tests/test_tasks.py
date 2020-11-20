from built_in_tasks.othertasks import LaserExperiment

import unittest
from tests.test_features import init_exp
import numpy as np

class TestOtherTasks(unittest.TestCase):

    def test_gen(self):
        powers, edges = LaserExperiment.pulse(10, [0.005], [1])
        self.assertCountEqual(powers, np.ones(10))
        for edge in edges:
            self.assertCountEqual(edge, [0, 0.005])

    def test_exp(self):
        exp = init_exp(LaserExperiment, [])
        exp.run()

if __name__ == '__main__':
    unittest.main()