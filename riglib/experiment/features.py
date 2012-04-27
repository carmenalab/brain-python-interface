import tempfile
import random
import traceback

import pygame

from riglib import calibrations

from . import traits

class RewardSystem(traits.HasTraits):
    '''Use the reward system during the reward phase'''
    def __init__(self, *args, **kwargs):
        from riglib import reward
        super(RewardSystem, self).__init__(*args, **kwargs)
        self.reward = reward.open()

    def _start_reward(self):
        if self.reward is not None:
            self.reward.reward(self.reward_time*1000.)
        super(RewardSystem, self)._start_reward()

class Autostart(traits.HasTraits):
    '''Automatically begins the trial from the wait state, with a random interval drawn from `rand_start`'''
    rand_start = traits.Array(value=(0.5, 2.), shape=(2,), desc="Start interval")

    def __init__(self, *args, **kwargs):
        self.pause = False
        super(Autostart, self).__init__(*args, **kwargs)

    def _start_wait(self):
        s, e = self.rand_start
        self.wait_time = random.random()*(e-s) + s
        super(Autostart, self)._start_wait()
        
    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause
    
    def _test_premature(self, ts):
        return self.event is not None

class Button(object):
    '''Adds the ability to respond to the button, as well as to keyboard responses'''
    def screen_init(self):
        super(Button, self).screen_init()
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def _get_event(self):
        btnmap = {1:1, 3:4}
        for btn in pygame.event.get(pygame.MOUSEBUTTONDOWN):
            if btn.button in btnmap:
                return btnmap[btn.button]

        return super(Button, self)._get_event()
    
    def _while_reward(self):
        super(Button, self)._while_reward()
        pygame.event.clear()
    
    def _while_penalty(self):
        #Clear out the button buffers
        super(Button, self)._while_penalty()
        pygame.event.clear()
    
    def _while_wait(self):
        super(Button, self)._while_wait()
        pygame.event.clear()

class IgnoreCorrectness(object):
    '''Allows any response to be correct, not just the one defined. Overrides for trialtypes'''
    def __init__(self, *args, **kwargs):
        super(IgnoreCorrectness, self).__init__(*args, **kwargs)
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


class AdaptiveGenerator(object):
    def __init__(self, *args, **kwargs):
        super(AdaptiveGenerator, self).__init__(*args, **kwargs)
        assert hasattr(self.gen, "correct"), "Must use adaptive generator!"

    def _start_reward(self):
        self.gen.correct()
        super(AdaptiveGenerator, self)._start_reward()
    
    def _start_incorrect(self):
        self.gen.incorrect()
        super(AdaptiveGenerator, self)._start_incorrect()







class EyeData(object):
    def __init__(self, *args, **kwargs):
        from riglib import shm
        self.eyedata = shm.EyeData()
        super(EyeData, self).__init__(*args, **kwargs)

    def run(self):
        self.eyedata.start()
        try:
            super(EyeData, self).run()
        finally:
            self.eyedata.pause()
            del self.eyedata
    
    def _start_None(self):
        self.eyedata.pause()
        self.eyefile = tempfile.mktemp()
        print "retrieving data from eyetracker..."
        self.eyedata.retrieve(self.eyefile)
        print "Done!"
        self.eyedata.stop()
        super(EyeData, self)._start_None()
    
    def set_state(self, state, **kwargs):
        self.eyedata.sendMsg(state)
        super(EyeData, self).set_state(state, **kwargs)

class CalibratedEyeData(EyeData):
    cal_profile = traits.Instance(calibrations.EyeProfile)

    def __init__(self, *args, **kwargs):
        from riglib import shm
        self.eyedata = shm.EyeData()
        self.eyedata.set_filter(self.cal_profile)
        super(CalibratedEyeData, self).__init__(*args, **kwargs)

class FixationStart(CalibratedEyeData):
    fixation_length = traits.Float(2., desc="Length of fixation required to start the task")
    fixation_dist = traits.Float(50., desc="Distance from center that is considered a broken fixation")

    def __init__(self, *args, **kwargs):
        super(FixationStart, self).__init__(*args, **kwargs)
        self.status['wait']['fixation_break'] = "wait"
        self.log_exclude.add(("wait", "fixation_break"))
    
    def _start_wait(self):
        self.eyedata.get()
        super(FixationStart, self)._start_wait()

    def _test_fixation_break(self, ts):
        return (np.sqrt((self.eyedata.get()**2).sum(1)) > self.fixation_dist).any()
    
    def _test_start_trial(self, ts):
        return ts > self.fixation_length

class SimulatedEyeData(EyeData):
    fixations = traits.Array(value=[(0,0), (-0.6,0.3), (0.6,0.3)], desc="Location of fixation points")
    fixation_len = traits.Float(0.5, desc="Length of a fixation")

    def __init__(self, *args, **kwargs):
        from riglib import shm
        super(SimulatedEyeData, self).__init__(*args, **kwargs)
        self.eyedata = shm.EyeSimulate(fixations=self.fixations, isi=self.fixation_len*1e3)


class MotionData(traits.HasTraits):
    marker_count = traits.Int(8, desc="Number of markers to return")

    def __init__(self, *args, **kwargs):
        from riglib import shm
        self.motiondata = shm.MotionData(marker_count=self.marker_count)
        super(MotionData, self).__init__(*args, **kwargs)

    def run(self):
        self.motiondata.start()
        try:
            super(MotionData, self).run()
        finally:
            self.motiondata.pause()
    
    def _start_None(self):
        self.motiondata.pause()
        self.motiondata.stop()
        super(MotionData, self)._start_None()

class MotionSimulate(traits.HasTraits):
    marker_count = traits.Int(8, desc="Number of markers to return")

    def __init__(self, *args, **kwargs):
        from riglib import shm
        self.motiondata = shm.MotionSimulate(marker_count=self.marker_count, 
            radius=(100,100,50), offset=(-150,0,0))
        super(MotionSimulate, self).__init__(*args, **kwargs)

    def run(self):
        self.motiondata.start()
        try:
            super(MotionSimulate, self).run()
        finally:
            self.motiondata.pause()
    
    def _start_None(self):
        self.motiondata.pause()
        self.motiondata.stop()
        super(MotionSimulate, self)._start_None()

class SaveHDF(object):
    '''Saves any associated MotionData and EyeData into an HDF5 file.'''
    def __init__(self, *args, **kwargs):
        super(SaveHDF, self).__init__(*args, **kwargs)

        import tempfile
        from riglib import datasink, hdfwriter
        self.h5file = tempfile.NamedTemporaryFile()
        self.sinks = datasink.sinks
        
        if isinstance(self, (MotionData, MotionSimulate)):
            self.sinks.register(self.motiondata)
        if isinstance(self, EyeData):
            self.sinks.register(self.eyedata)

        self.sinks.start(hdfwriter.HDFWriter, filename=self.h5file.name)
    
    def run(self):
        try:
            super(SaveHDF, self).run()
        finally:
            self.sinks.stop()

    def set_state(self, condition, **kwargs):
        for sink in self.sinks:
            for source in self.sinks.sources:
                sink.sendMsg(source.source, condition)

        super(SaveHDF, self).set_state(condition, **kwargs)
    
    def _start_None(self):
        self.sinks.stop()
        super(SaveHDF, self)._start_None()

class RelayPlexon(object):
    def __init__(self, *args, **kwargs):
        super(RelayPlexon, self).__init__(*args, **kwargs)

        import tempfile
        from riglib import datasink, nidaq
        self.sinks = datasink.sinks
        
        if isinstance(self, (MotionData, MotionSimulate)):
            self.sinks.register(self.motiondata)
        if isinstance(self, EyeData):
            self.sinks.register(self.eyedata)

        self.sinks.start(nidaq.Output)
    
    def run(self):
        try:
            super(RelayPlexon, self).run()
        finally:
            self.sinks.stop()

    def set_state(self, condition, **kwargs):
        for sink in self.sinks:
            for source in self.sinks.sources:
                sink.sendMsg(source.source, condition)

        super(RelayPlexon, self).set_state(condition, **kwargs)
    
    def _start_None(self):
        self.sinks.stop()
        super(RelayPlexon, self)._start_None()
