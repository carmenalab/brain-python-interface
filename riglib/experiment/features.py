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
    rand_start = traits.Tuple((0.5, 2.), desc="Start interval")

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







class EyeData(traits.HasTraits):
    '''Pulls data from the eyetracking system and make it available on self.eyedata'''
    def __init__(self, *args, **kwargs):
        from riglib import motiontracker, source
        self.eyedata = source.DataSource(eyetracker.System)
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
    '''Filters eyetracking data with a calibration profile'''
    cal_profile = traits.Instance(calibrations.EyeProfile)

    def __init__(self, *args, **kwargs):
        super(CalibratedEyeData, self).__init__(*args, **kwargs)
        self.eyedata.set_filter(self.cal_profile)

class FixationStart(CalibratedEyeData):
    '''Triggers the start_trial event whenever fixation exceeds *fixation_length*'''
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
    '''Simulate an eyetracking system using a series of fixations, with saccades interpolated'''
    fixations = traits.Array(value=[(0,0), (-0.6,0.3), (0.6,0.3)], desc="Location of fixation points")
    fixation_len = traits.Float(0.5, desc="Length of a fixation")

    def __init__(self, *args, **kwargs):
        from riglib import eyetracker, source
        self.eyedata = source.DataSource(eyetracker.Simulate)
        super(SimulatedEyeData, self).__init__(*args, **kwargs)



class MotionData(traits.HasTraits):
    '''Enable reading of raw motiontracker data from Phasespace system'''
    marker_count = traits.Int(8, desc="Number of markers to return")

    def __init__(self, *args, **kwargs):
        from riglib import motiontracker, source
        Motion = motiontracker.make_system(self.marker_count)
        self.motiondata = source.DataSource(Motion)
        super(MotionData, self).__init__(*args, **kwargs)

    def run(self):
        self.motiondata.start()
        try:
            super(MotionData, self).run()
        finally:
            self.motiondata.pause()
            self.motiondata.stop()

class MotionSimulate(traits.HasTraits):
    '''Simulate presence of raw motiontracking system using a randomized spatial function'''
    marker_count = traits.Int(8, desc="Number of markers to return")

    def __init__(self, *args, **kwargs):
        from riglib import motiontracker, source
        Motion = motiontracker.make_simulate(self.marker_count)
        self.motiondata = source.DataSource(Motion, radius=(100,100,50), offset=(-150,0,0))
        super(MotionSimulate, self).__init__(*args, **kwargs)

    def run(self):
        self.motiondata.start()
        try:
            super(MotionSimulate, self).run()
        finally:
            self.motiondata.pause()
            self.motiondata.stop()

class SpikeData(object):
    '''Stream neural spike data from the Plexon system'''
    marker_count = traits.Int(8, desc="Number of markers to return")
    spikebin_interval = traits.Float(100, desc="Milliseconds to bin over to generate the PSTH")
    plexon_channels = None

    def __init__(self, *args, **kwargs):
        from riglib import plexon, source
        self.neurondata = source.DataSource(plexon.Spikes, channels=self.plexon_channels)
        self.neurondata.filter = plexon.PSTHfilter(self.spikebin_interval)
        super(SpikeData, self).__init__(*args, **kwargs)

    def run(self):
        self.neurondata.start()
        try:
            super(SpikeData, self).run()
        finally:
            self.neurondata.stop()


class SinkRegister(object):
    '''Superclass for all features which contain data sinks -- registers the various sources'''
    def __init__(self, *args, **kwargs):
        from riglib import sink
        self.sinks = sink.sinks

        super(SinkRegister, self).__init__(*args, **kwargs)

        if isinstance(self, (MotionData, MotionSimulate)):
            self.sinks.register(self.motiondata)
        if isinstance(self, (EyeData, CalibratedEyeData, SimulatedEyeData)):
            self.sinks.register(self.eyedata)
        if isinstance(self, (SpikeData, SpikeSimulate)):
            self.sinks.register(self.neurondata)

class SaveHDF(SinkRegister):
    '''Saves any associated MotionData and EyeData into an HDF5 file.'''
    def __init__(self, *args, **kwargs):
        import tempfile
        from riglib import sink
        self.h5file = tempfile.NamedTemporaryFile()
        self.hdf = sink.sinks.start(self.hdf_class, filename=self.h5file.name)
        super(SaveHDF, self).__init__(*args, **kwargs)

    @property
    def hdf_class(self):
        from riglib import hdfwriter
        return hdfwriter.HDFWriter

    def run(self):
        try:
            super(SaveHDF, self).run()
        finally:
            self.hdf.stop()

    def set_state(self, condition, **kwargs):
        self.hdf.sendMsg(condition)
        super(SaveHDF, self).set_state(condition, **kwargs)

class RelayPlexon(SinkRegister):
    '''Sends the full data from eyetracking and motiontracking systems directly into Plexon'''
    def __init__(self, *args, **kwargs):
        from riglib import sink
        self.nidaq = sink.sinks.start(self.ni_out)
        super(RelayPlexon, self).__init__(*args, **kwargs)

    @property
    def ni_out(self):
        from riglib import nidaq
        return nidaq.SendAll
    
    def run(self):
        try:
            super(RelayPlexon, self).run()
        finally:
            self.nidaq.stop()

    def set_state(self, condition, **kwargs):
        self.nidaq.sendMsg(condition)
        super(RelayPlexon, self).set_state(condition, **kwargs)
        
class RelayPlexByte(RelayPlexon):
    '''Relays a single byte (0-255) as a row checksum for when a data packet arrives'''
    def __init__(self, *args, **kwargs):
        if not isinstance(self, SaveHDF):
            raise ValueError("RelayPlexByte feature only available with SaveHDF")
        super(RelayPlexByte, self).__init__(*args, **kwargs)

    @property
    def ni_out(self):
        from riglib import nidaq
        return nidaq.SendRowByte
