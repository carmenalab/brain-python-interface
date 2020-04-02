#!/usr/bin/python 
'''
Unit test for fixed BMI
'''
import unittest
import tasks
from tasks import performance
from tasks import generatorfunctions as genfns
import numpy as np
import testing

def test_block(idx, n_iter=None):
    te = performance._get_te(idx)
    if n_iter == None:
        n_iter = len(te.hdf.root.task)
        
    gen = genfns.sim_target_seq_generator_multi(8, 1000)
    task = tasks.BMIReconstruction(te, n_iter, gen)
    task.init()
        
    error = task.calc_recon_error(verbose=False)
    abs_max_error = np.max(np.abs(error))
    return abs_max_error

class TestFixedDecoderBMI(unittest.TestCase):
    def test_ppf_fixed(self):
        print("Testing fixed PPF block")
        abs_max_error = test_block(2295)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)    

    def test_kf_fixed(self):
        print("Testing fixed KF block")
        abs_max_error = test_block(2908)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)    
        
if __name__ == '__main__':
    unittest.main()

