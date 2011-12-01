import random

from __init__ import options
from .. import button

class Autostart(object):
    def _start_wait(self):
        s, e = options['rand_start']
        self.wait_time = random.random()*(e-s) + s

    def _test_start_trial(self, ts):
        return ts > self.wait_time

class Button(object):
    def __init__(self):
        super(Button, self).__init__()
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
    def _test_correct(self, ts):
        return self.event is not None

    def _test_incorrect(self, ts):
        return False