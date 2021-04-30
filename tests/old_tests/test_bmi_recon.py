#!/usr/bin/python 
'''
BMI reconstruction test cases. Any modificaitons to BMI code should pass all these tests
'''
import unittest
import tasks
from analysis import performance
from tasks import generatorfunctions as genfns
import numpy as np 

def bmi_block_reconstruction_error(idx, cls=tasks.BMIReconstruction, n_iter=None):
    te = performance._get_te(idx)
    if n_iter == None or n_iter == -1: n_iter = len(te.hdf.root.task)
        
    gen = genfns.sim_target_seq_generator_multi(8, 1000)
    task = cls(te, n_iter, gen)
    task.init()
        
    error = task.calc_recon_error(verbose=False)
    abs_max_error = np.max(np.abs(error))
    return abs_max_error


class TestFixedPPF(unittest.TestCase):
    def runTest(self):
        print("Testing fixed PPF block")
        abs_max_error = bmi_block_reconstruction_error(2295, n_iter=n_iter)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)    

class TestFixedKF(unittest.TestCase):
    def runTest(self):
        print("Testing fixed KF block")
        abs_max_error = bmi_block_reconstruction_error(2908, n_iter=n_iter)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)

class TestPPFCLDA(unittest.TestCase):
    def runTest(self):
        print("Testing adapting PPF block")
        abs_max_error = bmi_block_reconstruction_error(2306, cls=tasks.ContCLDARecon, n_iter=n_iter)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)        

class TestRML(unittest.TestCase):
    def runTest(self):
        print("Testing RML block")
        abs_max_error = bmi_block_reconstruction_error(3133, cls=tasks.KFRMLRecon, n_iter=n_iter)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)

class TestRMLWithAssist(unittest.TestCase):
    def runTest(self):
        print("Testing RML block with assist")
        raise NotImplementedError("ID Number is wrong! find a correct block")
        abs_max_error = bmi_block_reconstruction_error(3040, cls=tasks.KFRMLRecon, n_iter=n_iter)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)

class TestRMLCGWithAssist(unittest.TestCase):
    def runTest(self):
        print("Testing RML block with assist")
        abs_max_error = bmi_block_reconstruction_error(5270, cls=tasks.KFRMLCGRecon, n_iter=n_iter)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)

class TestJointRML(unittest.TestCase):
    def runTest(self):
        print("Testing Joint RML")
        abs_max_error = bmi_block_reconstruction_error(3040, cls=tasks.KFRMLJointRecon, n_iter=n_iter)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)

class TestJointRMLWithAssist(unittest.TestCase):
    def runTest(self):
        print("Testing Joint RML block with assist")
        abs_max_error = bmi_block_reconstruction_error(3088, cls=tasks.KFRMLJointRecon, n_iter=n_iter)
        print(abs_max_error)
        self.assertTrue(abs_max_error < 1e-10)



import argparse
parser = argparse.ArgumentParser(description='Analyze perf correlates of KF plant properties')
parser.add_argument('--n_iter', help='', type=int, action="store", default=-1)
args = parser.parse_args()

n_iter = args.n_iter
tests = []
# tests.append(TestFixedPPF())
# tests.append(TestFixedKF())
# tests.append(TestPPFCLDA())
# tests.append(TestRML)
# tests.append(TestRMLWithAssist())
# tests.append(TestJointRMLWithAssist())


te = performance._get_te(3040)

#tests = [TestJointRML(), TestJointRMLWithAssist()]
#tests = [TestFixedPPF(), TestPPFCLDA()]

#tests = [TestPPFCLDA(), TestRML()]

# tests = [TestRMLCGWithAssist()]
tests = [TestPPFCLDA()]

test_suite = unittest.TestSuite(tests)
unittest.TextTestRunner(verbosity=2).run(test_suite)
