import unittest, os

from tests.test_decoders import TestLinDec
from tests.test_features import TestKeyboardControl, TestMouseControl
test_classes = [
    TestLinDec,
    TestKeyboardControl,
    TestMouseControl,
]
 
suite = unittest.TestSuite()

for cls in test_classes:
    suite.addTest(unittest.makeSuite(cls))      

runner = unittest.TextTestRunner()
runner_output = runner.run(suite)
