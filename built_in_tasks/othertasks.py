''' Tasks which don't include any visuals or bmi, such as laser-only or camera-only tasks'''

from riglib.experiment import Experiment, Sequence
from features.laser_features import LaserTrials

class LaserExperiment(Sequence, LaserTrials):
    
    status = dict(
        wait = dict(start_trial="trial", stop=None),
        trial = dict(end_trial="wait"),
    )
    
    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause
    
    def _test_end_trial(self, ts):
        return ts > self.laser_duration

class WhiteMatterCamera(Experiment):
    pass