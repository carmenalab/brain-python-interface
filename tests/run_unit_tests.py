import unittest, os

from tests.test_decoders import TestLinDec
test_classes = [
    TestLinDec
]
 
suite = unittest.TestSuite()

for cls in test_classes:
    suite.addTest(unittest.makeSuite(cls))      

runner = unittest.TextTestRunner()
runner_output = runner.run(suite)
