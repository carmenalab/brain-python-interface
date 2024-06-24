import numpy as np
import unittest
from reqlib import swreq
from requirements import *

from features.hdf_features import SaveHDF
import time
import tables

from riglib import experiment
from riglib import sink
import mocks
import os

from riglib.experiment.mocks import MockSequenceWithTraits



class TestTraits(unittest.TestCase):
    def setUp(self):
        Exp = MockSequenceWithTraits
        params = Exp.get_params()
        self.params = params 

    def test_options_trait(self):
        """Test that the options trait has the drop-down options available"""
        self.assertEqual(self.params['options_trait']['options'], ['option1', 'option2'])

    def test_trait_order(self):
        """Test that the order of the traits is in the order specified during declaration of the task"""
        self.assertEqual(list(self.params.keys()), ['options_trait', 'float_trait', 'session_length'])
        

if __name__ == '__main__':
    unittest.main()