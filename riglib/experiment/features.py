import random

from . import traits
from .. import button

class Autostart(traits.HasTraits):
    rand_start = traits.Tuple((1, 10))

    def _start_wait(self):
        s, e = self.rand_start
        self.wait_time = random.random()*(e-s) + s
        
    def _test_start_trial(self, ts):
        return ts > self.wait_time
    
    def _test_premature(self, ts):
        return self.event is not None

class Button(object):
    def __init__(self, **kwargs):
        super(Button, self).__init__(**kwargs)
        try:
            self.button = button.Button()
        except:
            print "Cannot find ftdi button"
            self.button = None
    
    def _get_event(self):
        return (self.button is not None and self.button.pressed()) or \
                super(Button, self)._get_event()

class ButtonOnly(Button):
    def _get_event(self):
        assert self.button is not None
        return self.button.pressed()

class IgnoreCorrectness(object):
    def __init__(self, **kwargs):
        super(IgnoreCorrectness, self).__init__(**kwargs)
        if hasattr(self, "trial_types"):
            for ttype in self.trial_types:
                del self.status[ttype]["%s_correct"%ttype]
                del self.status[ttype]["%s_incorrect"%ttype]
                self.status[ttype]["correct"] = "reward"
                self.status[ttype]["incorrect"] = "penalty"

    def _test_correct(self, ts):
        return self.event is not None

    def _test_incorrect(self, ts):
        return False