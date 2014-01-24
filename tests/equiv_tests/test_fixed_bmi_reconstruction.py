#!/usr/bin/python 
'''
Unit test for fixed BMI
'''
import unittest
import tasks
from tasks import performance
from tasks import generatorfunctions as genfns
import numpy as np

class TestFixedDecoderBMI(unittest.TestCase):
    def test_ppf_fixed(self):
        
        idx = 2295
        te = performance._get_te(idx)
        n_iter = len(te.hdf.root.task)
        
        gen = genfns.sim_target_seq_generator_multi(8, 1000)
        task = tasks.BMIReconstruction(te, n_iter, gen)
        task.init()
        
        error = task.calc_recon_error(verbose=False)
        abs_max_error = np.max(np.abs(error))
        self.assertTrue(abs_max_error < 1e-10)

        
if __name__ == '__main__':
    unittest.main()

