import time
import random
import pygame

from riglib import experiment, gendots, reward, options
from visual_response import Dots

class Depth(Dots):
    def _test_correct(self, ts):
        return (not self._popout and ts > options['ignore_time']) or (self._popout and self.event is not None)

    def _test_incorrect(self, ts):
        return (self._popout and ts > options['ignore_time']) or (not self._popout and self.event is not None)
    
if __name__ == "__main__":
    exp = Depth()
    exp.run()