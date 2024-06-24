import unittest, os

from tests.test_features import TestKeyboardControl, TestMouseControl, TestMouseJoystick
from tests.test_tasks import TestOtherTasks
test_classes = [
    TestKeyboardControl,
    TestMouseControl,
    TestMouseJoystick,
    TestOtherTasks,
]
 
suite = unittest.TestSuite()

for cls in test_classes:
    suite.addTest(unittest.makeSuite(cls))      

runner = unittest.TextTestRunner()
runner_output = runner.run(suite)
