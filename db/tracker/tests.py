"""
This file demonstrates writing tests using the unittest module. These will pass
when you run "manage.py test".

Replace this with more appropriate tests for your application.
"""

from django.test import TestCase
import json

class SimpleTest(TestCase):
    def test_basic_addition(self):
        """
        Tests that 1 + 1 always equals 2.
        """
        self.assertEqual(1 + 1, 2)

def test_make_params():
    from riglib import experiment
    from riglib.experiment import features
    from tasks import manualcontrol

    jsdesc = dict()
    Exp = experiment.make(manualcontrol.TargetDirection, (features.Autostart, features.MotionSimulate, features.SaveHDF))
    ctraits = Exp.class_traits()
    for trait in Exp.class_editable_traits():
        varname = dict()
        varname['type'] = ctraits[trait].trait_type.__class__.__name__
        varname['default'] = ctraits[trait].default
        varname['desc'] = ctraits[trait].desc
        jsdesc[trait] = varname

    return json.dumps(jsdesc)