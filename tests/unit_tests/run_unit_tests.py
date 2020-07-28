import unittest, os

from test_riglib_source import TestDataSourceSystem
from test_riglib_bmi import TestKalmanFilter, TestKFDecoder, TestZeroVelocityGoal, \
    TestGaussianState, TestNullAccumulator, TestRectAccumulator
from test_riglib_experiment import TestLogExperiment, TestSequence
from test_riglib_hdfwriter import TestHDFWriter
from test_feature_savehdf import TestSaveHDF
from test_mixin_features import TestTaskWithFeatures
from test_riglib_traits import TestTraits
from test_built_in_vfb_task import TestVisualFeedback

from requirements import *

test_classes = [
    TestKalmanFilter, 
    TestDataSourceSystem, TestKFDecoder, TestLogExperiment, 
    TestSequence, TestHDFWriter, TestZeroVelocityGoal, TestNullAccumulator,
    TestSaveHDF, TestTraits, TestTaskWithFeatures, TestVisualFeedback,
]

import reqlib
 
suite = unittest.TestSuite()
result = unittest.TestResult()

for cls in test_classes:
    suite.addTest(unittest.makeSuite(cls))      

suite_tests = suite._tests.copy() # making a copy because this list gets destroyed in the python 3 implementation

runner = unittest.TextTestRunner()
runner_output = runner.run(suite)

git_hash = os.popen("git rev-parse HEAD").readlines()[0].strip()
runner_output_text = "Git hash: %s\nAll tests pass? %s\n\n" % (git_hash, str(runner_output.wasSuccessful()))

# import ipdb; ipdb.set_trace()
reqlib.generate_traceability_matrix(all_requirements, suite_tests, runner_output_text)
