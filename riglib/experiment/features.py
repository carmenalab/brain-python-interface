'''
Code for the main "features" that can be used to extend and customize a 
task/experiment by multiple inheritance.
'''

import time
import tempfile
import random
import traceback
import numpy as np
import fnmatch
import os

from riglib import calibrations, bmi

from riglib.bmi import extractor

from . import traits, experiment

import os
import subprocess

import time

###### CONSTANTS
sec_per_min = 60

class RewardSystem(traits.HasTraits):
    '''Use the reward system during the reward phase'''
    def __init__(self, *args, **kwargs):
        from riglib import reward
        super(RewardSystem, self).__init__(*args, **kwargs)
        self.reward = reward.open()

    def _start_reward(self):
        if self.reward is not None:
            self.reward.reward(self.reward_time*1000.)
            self.reportstats['Reward #'] = self.reportstats['Reward #'] + 1
        super(RewardSystem, self)._start_reward()

class TTLReward(traits.HasTraits):
    '''During the reward phase, send a timed TTL pulse to the reward system'''
    def __init__(self, *args, **kwargs):
        '''
        Constructor for TTLReward

        Parameters
        ----------
        pulse_device: string
            Path to the NIDAQ device used to generate the solenoid pulse
        args, kwargs: optional positional and keyword arguments to be passed to parent constructor
            None necessary

        Returns
        -------
        TTLReward instance
        '''
        import comedi
        self.com = comedi.comedi_open('/dev/comedi0')
        super(TTLReward, self).__init__(*args, **kwargs)

    def _start_reward(self):
        '''
        At the start of the reward state, turn on the solenoid

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        import comedi
        super(TTLReward, self)._start_reward()
        self.reportstats['Reward #'] = self.reportstats['Reward #'] + 1
        subdevice = 0
        write_mask = 0x800000
        val = 0x800000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, val, base_channel)
        self.reward_start = self.get_time() - self.start_time

    def _test_reward_end(self, ts):
        return (ts - self.reward_start) > self.reward_time

    def _end_reward(self):
        '''
        After the reward state has elapsed, turn off the solenoid

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        import comedi
        subdevice = 0
        write_mask = 0x800000
        val = 0x000000
        base_channel = 0
        comedi.comedi_dio_bitfield2(self.com, subdevice, write_mask, 0x000000, base_channel)

class JuiceLogging(traits.HasTraits):
    '''
    Save screenshots of the juice camera and link them to the task entry that has been created
    '''
    def cleanup(self, database, saveid, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(JuiceLogging, self).cleanup(database, saveid, **kwargs)

        ## Remove the old screenshot, if any
        fname = '/storage/temp/_juice_logging_temp.png'
        subprocess.call(['rm', fname])

        ## Use the script to run the snapshot
        subprocess.call(['camera_snapshot.sh', fname])
        # os.subprocess('camera_snapshot.sh %s' % fname)

        ## Wait for a second to make sure the file is created (TODO should really be a timed loop!)
        time.sleep(1)

        ## Link the image to the database
        database.save_data(fname, 'juice_log', saveid)


class Autostart(traits.HasTraits):
    '''Automatically begins the trial from the wait state, with a random interval drawn from `rand_start`'''
    rand_start = traits.Tuple((0.5, 2.), desc="Start interval")

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.pause = False
        super(Autostart, self).__init__(*args, **kwargs)

    def _start_wait(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        s, e = self.rand_start
        self.wait_time = random.random()*(e-s) + s
        super(Autostart, self)._start_wait()
        
    def _test_start_trial(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return ts > self.wait_time and not self.pause
    
    def _test_premature(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return self.event is not None

class Button(object):
    '''Adds the ability to respond to the button, as well as to keyboard responses'''
    def screen_init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(Button, self).screen_init()
        import pygame
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def _get_event(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        import pygame
        btnmap = {1:1, 3:4}
        for btn in pygame.event.get(pygame.MOUSEBUTTONDOWN):
            if btn.button in btnmap:
                return btnmap[btn.button]

        return super(Button, self)._get_event()
    
    def _while_reward(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(Button, self)._while_reward()
        import pygame
        pygame.event.clear()
    
    def _while_penalty(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        #Clear out the button buffers
        super(Button, self)._while_penalty()
        import pygame
        pygame.event.clear()
    
    def _while_wait(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(Button, self)._while_wait()
        import pygame
        pygame.event.clear()

class IgnoreCorrectness(object):
    '''Allows any response to be correct, not just the one defined. Overrides for trialtypes'''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(IgnoreCorrectness, self).__init__(*args, **kwargs)
        if hasattr(self, "trial_types"):
            for ttype in self.trial_types:
                del self.status[ttype]["%s_correct"%ttype]
                del self.status[ttype]["%s_incorrect"%ttype]
                self.status[ttype]["correct"] = "reward"
                self.status[ttype]["incorrect"] = "penalty"

    def _test_correct(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return self.event is not None

    def _test_incorrect(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return False

class AdaptiveGenerator(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(AdaptiveGenerator, self).__init__(*args, **kwargs)
        assert hasattr(self.gen, "correct"), "Must use adaptive generator!"

    def _start_reward(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.gen.correct()
        super(AdaptiveGenerator, self)._start_reward()
    
    def _start_incorrect(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.gen.incorrect()
        super(AdaptiveGenerator, self)._start_incorrect()

class Joystick(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import source, phidgets
        System = phidgets.make(2, 1)
        self.joystick = source.DataSource(System)
        super(Joystick, self).init()
        if isinstance(self, SaveHDF):
            self.add_dtype('joystick_sensor_vals', 'f8', (2,))

    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.joystick.start()
        try:
            super(Joystick, self).run()
        finally:
            self.joystick.stop()

    def join(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.joystick.join()
        super(Joystick, self).join()

class DualJoystick(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import source, phidgets
        System = phidgets.make(4, 1)
        self.dualjoystick = source.DataSource(System)
        super(DualJoystick, self).init()

    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.dualjoystick.start()
        try:
            super(DualJoystick, self).run()
        finally:
            self.dualjoystick.stop()

    def join(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.dualjoystick.join()
        super(DualJoystick, self).join()

########################################################################################################
# Eyetracker datasources
########################################################################################################
class EyeData(traits.HasTraits):
    '''Pulls data from the eyetracking system and make it available on self.eyedata'''

    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import source
        src, ekw = self.eye_source
        f = open('/home/helene/code/bmi3d/log/eyetracker', 'a')
        self.eyedata = source.DataSource(src, **ekw)
        f.write('instantiated source\n')
        super(EyeData, self).init()
        f.close()
    
    @property
    def eye_source(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import eyetracker
        return eyetracker.System, dict()

    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        f = open('/home/helene/code/bmi3d/log/eyetracker', 'a')
        self.eyedata.start()
        f.write('started eyedata\n')
        f.close()
        try:
            super(EyeData, self).run()
        finally:
            self.eyedata.stop()
    
    def join(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.eyedata.join()
        super(EyeData, self).join()
    
    def _start_None(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.eyedata.pause()
        self.eyefile = tempfile.mktemp()
        print "retrieving data from eyetracker..."
        self.eyedata.retrieve(self.eyefile)
        print "Done!"
        self.eyedata.stop()
        super(EyeData, self)._start_None()
    
    def set_state(self, state, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.eyedata.sendMsg(state)
        super(EyeData, self).set_state(state, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(EyeData, self).cleanup(database, saveid, **kwargs)
        database.save_data(self.eyefile, "eyetracker", saveid)

class SimulatedEyeData(EyeData):
    '''Simulate an eyetracking system using a series of fixations, with saccades interpolated'''
    fixations = traits.Array(value=[(0,0), (-0.6,0.3), (0.6,0.3)], desc="Location of fixation points")
    fixation_len = traits.Float(0.5, desc="Length of a fixation")

    @property
    def eye_source(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import eyetracker
        return eyetracker.Simulate, dict(fixations=fixations, fixation_len=fixation_len)

class CalibratedEyeData(EyeData):
    '''Filters eyetracking data with a calibration profile'''
    cal_profile = traits.Instance(calibrations.EyeProfile)

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(CalibratedEyeData, self).__init__(*args, **kwargs)
        self.eyedata.set_filter(self.cal_profile)

class FixationStart(CalibratedEyeData):
    '''Triggers the start_trial event whenever fixation exceeds *fixation_length*'''
    fixation_length = traits.Float(2., desc="Length of fixation required to start the task")
    fixation_dist = traits.Float(50., desc="Distance from center that is considered a broken fixation")

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(FixationStart, self).__init__(*args, **kwargs)
        self.status['wait']['fixation_break'] = "wait"
        self.log_exclude.add(("wait", "fixation_break"))
    
    def _start_wait(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.eyedata.get()
        super(FixationStart, self)._start_wait()

    def _test_fixation_break(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return (np.sqrt((self.eyedata.get()**2).sum(1)) > self.fixation_dist).any()
    
    def _test_start_trial(self, ts):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        return ts > self.fixation_length

########################################################################################################
# Phasespace datasources
########################################################################################################
class MotionData(traits.HasTraits):
    '''Enable reading of raw motiontracker data from Phasespace system'''
    marker_count = traits.Int(8, desc="Number of markers to return")

    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import source
        src, mkw = self.motion_source
        self.motiondata = source.DataSource(src, **mkw)
        super(MotionData, self).init()
    
    @property
    def motion_source(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import motiontracker
        return motiontracker.make(self.marker_count), dict()

    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.motiondata.start()
        try:
            super(MotionData, self).run()
        finally:
            self.motiondata.stop()
    
    def join(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.motiondata.join()
        super(MotionData, self).join()
    
    def _start_None(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.motiondata.stop()
        super(MotionData, self)._start_None()

class MotionSimulate(MotionData):
    '''Simulate presence of raw motiontracking system using a randomized spatial function'''
   
    @property
    def motion_source(self):
        from riglib import motiontracker
        cls = motiontracker.make(self.marker_count, cls=motiontracker.Simulate)
        return cls, dict(radius=(100,100,50), offset=(-150,0,0))

class MotionAutoAlign(MotionData):
    '''Creates an auto-aligning motion tracker, for use with the 6-point alignment system'''
    autoalign = traits.Instance(calibrations.AutoAlign)
    
    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(MotionAutoAlign, self).init()
        self.motiondata.filter = self.autoalign

    @property
    def motion_source(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import motiontracker
        cls = motiontracker.make(self.marker_count, cls=motiontracker.AligningSystem)
        return cls, dict()


class BlackrockData(traits.HasTraits):
    '''Stream Blackrock neural data.'''
    blackrock_channels = None

    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import blackrock, source

        if 'spike' in self.decoder.extractor_cls.feature_type:  # e.g., 'spike_counts'
            self.neurondata = source.DataSource(blackrock.Spikes, channels=self.blackrock_channels, send_data_to_sink_manager=False)
        elif 'lfp' in self.decoder.extractor_cls.feature_type:  # e.g., 'lfp_power'
            self.neurondata = source.MultiChanDataSource(blackrock.LFP, channels=self.blackrock_channels, send_data_to_sink_manager=True)
        else:
            raise Exception("Unknown extractor class, unable to create data source object!")

        from riglib import sink
        sink.sinks.register(self.neurondata)

        super(BlackrockData, self).init()

    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.neurondata.start()
        try:
            super(BlackrockData, self).run()
        finally:
            self.neurondata.stop()

class BlackrockBMI(BlackrockData):
    '''Filters neural data from the Blackrock system through a BMI.'''
    decoder = traits.Instance(bmi.Decoder)

    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        print "init bmi"
        self.blackrock_channels = self.decoder.units[:,0]
        super(BlackrockBMI, self).init()


class BrainAmpData(traits.HasTraits):
    '''Stream BrainAmp neural data.'''

    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import brainamp, source

        self.emgdata = source.MultiChanDataSource(brainamp.EMG, channels=channels)

        try:
            super(BrainAmpData, self).init()
        except:
            print "BrainAmpData: running without a task"

    def run(self):
        self.emgdata.start()



#*******************************************************************************************************
# Data Sinks
#*******************************************************************************************************
class SinkRegister(object):
    '''Superclass for all features which contain data sinks -- registers the various sources'''
    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import sink
        self.sinks = sink.sinks

        # Run the rest of the .init() functions of the custom experiment class
        # NOTE: this MUST happen before the rest of the code executes. Otherwise,
        # the dtype used to determine the task data attributes to be stored
        # to the HDF file will be incorrect/incomplete
        super(SinkRegister, self).init()

        if isinstance(self, (MotionData, MotionSimulate)):
            self.sinks.register(self.motiondata)
        if isinstance(self, (EyeData, CalibratedEyeData, SimulatedEyeData)):
            self.sinks.register(self.eyedata)
        if isinstance(self, Joystick):
            self.sinks.register(self.joystick)

        # Register sink for task data
        try:
            self.dtype = np.dtype(self.dtype)
            self.sinks.register("task", self.dtype)
            self.task_data = np.zeros((1,), dtype=self.dtype)
        except:
            self.task_data = None

    def _cycle(self):
        ''' Docstring '''
        super(SinkRegister, self)._cycle()
        if self.task_data is not None:
            self.sinks.send("task", self.task_data)
        
class SaveHDF(SinkRegister):
    '''
    Saves data from registered sources into tables in an HDF file
    '''
    def init(self):
        ''' Docstring '''
        import tempfile
        from riglib import sink
        self.h5file = tempfile.NamedTemporaryFile()
        self.hdf = sink.sinks.start(self.hdf_class, filename=self.h5file.name)
        super(SaveHDF, self).init()

    @property
    def hdf_class(self):
        ''' Docstring '''
        from riglib import hdfwriter
        return hdfwriter.HDFWriter

    def run(self):
        ''' Docstring '''
        try:
            super(SaveHDF, self).run()
        finally:
            self.hdf.stop()
    
    def join(self):
        ''' Docstring '''
        self.hdf.join()
        super(SaveHDF, self).join()

    def set_state(self, condition, **kwargs):
        '''
        Save task state transitions to HDF

        Parameters
        ----------
        condition: string
            Name of new state to transition into. The state name must be a key in the 'status' dictionary attribute of the task

        Returns
        -------
        None
        '''
        self.hdf.sendMsg(condition)
        super(SaveHDF, self).set_state(condition, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(SaveHDF, self).cleanup(database, saveid, **kwargs)
        print "#################%s"%self.h5file.name
        try:
            self.cleanup_hdf()
        except:
            print "cleanup error!!!!!!!!!!!!!!!!!!!!"
            import traceback
            traceback.print_exc()

        database.save_data(self.h5file.name, "hdf", saveid)

########################################################################################################
# Plexon features
########################################################################################################
class PlexonData(traits.HasTraits):
    '''Stream Plexon neural data'''
    plexon_channels = None

    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import plexon, source

        if hasattr(self.decoder, 'extractor_cls'):
            if 'spike' in self.decoder.extractor_cls.feature_type:  # e.g., 'spike_counts'
                self.neurondata = source.DataSource(plexon.Spikes, channels=self.plexon_channels)
            elif 'lfp' in self.decoder.extractor_cls.feature_type:  # e.g., 'lfp_power'
                self.neurondata = source.MultiChanDataSource(plexon.LFP, channels=self.plexon_channels)
            elif 'emg' in self.decoder.extractor_cls.feature_type:  # e.g., 'emg_amplitude'
                self.neurondata = source.MultiChanDataSource(plexon.Aux, channels=self.plexon_channels)
            else:
                raise Exception("Unknown extractor class, unable to create data source object!")
        else:
            # if using an older decoder that doesn't have extractor_cls (and 
            # extractor_kwargs) as attributes, then just create a DataSource 
            # with plexon.Spikes by default
            self.neurondata = source.DataSource(plexon.Spikes, channels=self.plexon_channels)

        super(PlexonData, self).init()

    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.neurondata.start()
        try:
            super(PlexonData, self).run()
        finally:
            self.neurondata.stop()

class PlexonBMI(PlexonData):
    '''
    Special case of PlexonData which specifies a subset of channels to stream, i.e., the ones used by the Decoder
    May not be available for all recording systems. 
    '''
    decoder = traits.Instance(bmi.Decoder)

    def init(self):
        print "init bmi"
        self.plexon_channels = self.decoder.units[:,0]
        super(PlexonBMI, self).init()

class SpikeSimulate(object):
    '''
    Docstring

    Parameters
    ----------

    Returns
    -------
    '''
    pass

class RelayPlexon(SinkRegister):
    '''
    Sends the full data from eyetracking and motiontracking systems directly into Plexon
    '''
    def init(self):
        ''' Docstring '''
        from riglib import sink
        print "self.ni_out()", self.ni_out()
        self.nidaq = sink.sinks.start(self.ni_out)
        print "self.nidaq", self.nidaq
        super(RelayPlexon, self).init()

    @property
    def ni_out(self):
        ''' Docstring '''
        from riglib import nidaq
        print 'nidaq.SendAll', nidaq.SendAll
        return nidaq.SendAll

    @property
    def plexfile(self):
        '''
        Calculates the plexon file that's most likely associated with the current task
        based on the time at which the task ended and the "last modified" time of the 
        plexon files located at /storage/plexon/
        '''
        import os, sys, glob, time
        if len(self.event_log) < 1:
            return None
        
        start = self.event_log[-1][2]
        files = "/storage/plexon/*.plx"
        files = sorted(glob.glob(files), key=lambda f: abs(os.stat(f).st_mtime - start))
        
        if len(files) > 0:
            tdiff = os.stat(files[0]).st_mtime - start
            if abs(tdiff) < sec_per_min:
                 return files[0]
    
    def run(self):
        ''' Docstring '''
        try:
            super(RelayPlexon, self).run()
        finally:
            self.nidaq.stop()

    def set_state(self, condition, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.nidaq.sendMsg(condition)
        super(RelayPlexon, self).set_state(condition, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(RelayPlexon, self).cleanup(database, saveid, **kwargs)
        time.sleep(2)
        if self.plexfile is not None:
            database.save_data(self.plexfile, "plexon", saveid, True, False)
        
class RelayPlexByte(RelayPlexon):
    '''
    Relays a single byte (0-255) to synchronize the rows of the HDF table(s) with the plexon recording clock.
    '''
    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if not isinstance(self, SaveHDF):
            raise ValueError("RelayPlexByte feature only available with SaveHDF")
        super(RelayPlexByte, self).init()

    @property
    def ni_out(self):
        '''
        see documentation for RelayPlexon.ni_out 
        '''
        from riglib import nidaq
        return nidaq.SendRowByte


class RelayBlackrock(SinkRegister):
    '''Sends full data directly into the Blackrock system.'''
    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import sink
        print "self.ni_out()", self.ni_out()
        self.nidaq = sink.sinks.start(self.ni_out)
        print "self.nidaq", self.nidaq
        super(RelayBlackrock, self).init()

    @property
    def ni_out(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import nidaq
        print 'nidaq.SendAll', nidaq.SendAll
        return nidaq.SendAll

    @property    
    def blackrockfiles(self):
        '''Finds the blackrock files --- .nev and .nsx (.ns1 through .ns6) 
        --- that are most likely associated with the current task based on 
        the time at which the task ended and the "last modified" time of the 
        files located at /storage/blackrock/.
        '''
        import os, sys, glob, time
        if len(self.event_log) < 1:
            return None

        start = self.event_log[-1][2]
        
        files = []
        for ext in ['.nev', '.ns1', '.ns2', '.ns3', '.ns4', '.ns5', '.ns6']:
            pattern = "/storage/blackrock/*" + ext

            #matches = sorted(glob.glob(pattern), key=lambda f: abs(os.stat(f).st_mtime - start))
            matches = []
            for root, dirnames, filenames in os.walk('/storage/blackrock'):
                for filename in fnmatch.filter(filenames, '*' + ext):
                    matches.append(os.path.join(root, filename))

            sorted_matches = sorted(matches, key=lambda f: abs(os.stat(f).st_mtime - start))

            if len(sorted_matches) > 0:
                tdiff = os.stat(sorted_matches[0]).st_mtime - start
                if abs(tdiff) < 60:
                     files.append(sorted_matches[0])

        return files
    
    def run(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        try:
            super(RelayBlackrock, self).run()
        finally:
            self.nidaq.stop()

    def set_state(self, condition, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.nidaq.sendMsg(condition)
        super(RelayBlackrock, self).set_state(condition, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(RelayBlackrock, self).cleanup(database, saveid, **kwargs)
        time.sleep(2)
        for file_ in self.blackrockfiles:
            suffix = file_[-3:]  # e.g., 'nev', 'ns3', etc.
            # database.save_data(file_, "blackrock", saveid, True, False, custom_suffix=suffix)
            database.save_data(file_, "blackrock", saveid, False, False, suffix)
        
class RelayBlackrockByte(RelayBlackrock):
    '''Relays a single byte (0-255) as a row checksum for when a data packet arrives.'''
    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if not isinstance(self, SaveHDF):
            raise ValueError("RelayBlackrockByte feature only available with SaveHDF")
        super(RelayBlackrockByte, self).init()

    @property
    def ni_out(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        from riglib import nidaq
        return nidaq.SendRowByte

######################
## Simulation Features
######################
class SimHDF(object):
    '''
    An interface-compatbile HDF for simulations which do not require saving an
    HDF file
    '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(SimHDF, self).__init__(*args, **kwargs)
        from collections import defaultdict
        self.data = defaultdict(list)
        self.task_data = dict()

    def sendMsg(self, msg):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        pass

    def __setitem__(self, key, value):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.data[key].append(value)

class SimTime(object):
    ''' Docstring '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(SimTime, self).__init__(*args, **kwargs)
        self.start_time = 0.

    def get_time(self):
        ''' Docstring '''
        try:
            return self.cycle_count * 1./self.fps
        except:
            # loop_counter has not been initialized yet, return 0
            return 0

########################################################################################################
# Decoder/BMISystem add-ons
########################################################################################################
class NormFiringRates(traits.HasTraits):
    ''' Docstring '''
    
    norm_time = traits.Float(120., desc="Number of seconds to use for mean and SD estimate")

    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        raise NotImplementedError("This feature is extremely depricated and probably does not work properly anymore.")
        super(NormFiringRates, self).__init__(*args, **kwargs)
        import time
        self.starttime = time.time()
        self.elapsedtime=0
        self.count=0
        self.mFR = None
        self.mFR2 = None
        self.sdFR = None
        self.updated=False

    def update_fr_vals(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        raise NotImplementedError("decoder.bin_spikes no longer exists, use the feature extractor instead")
        if self.elapsedtime>1.:
            bin = self.decoder.bin_spikes(self.neurondata.get(all=True).copy())
            self.count +=1
            if self.count == 1:
                sz = len(bin)
                self.mFR=np.zeros([sz])
                self.mFR2=np.zeros([sz])
            delta = bin - self.mFR
            self.mFR = self.mFR + delta/self.count
            self.mFR2 = self.mFR2 + delta*(bin - self.mFR)

    def update_cursor(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.elapsedtime = time.time()-self.starttime
        if self.elapsedtime<self.norm_time:
            self.update_fr_vals()
        elif not self.updated:
            self.sdFR = np.sqrt(self.mFR2/(self.count-1))
            self.decoder.init_zscore(self.mFR,self.sdFR)
            self.hdf.sendMsg("baseline_norm")
            self.updated=True
            print "Updated session mean and SD."
            self.hdf.sendAttr("task", "session_mFR", self.mFR)
            self.hdf.sendAttr("task", "session_sdFR", self.sdFR)

        super(NormFiringRates, self).update_cursor()

class LinearlyDecreasingAttribute(traits.HasTraits):
    ''' Docstring '''
    def __init__(self, *args, **kwargs):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        assert isinstance(self, experiment.Experiment)
        super(LinearlyDecreasingAttribute, self).__init__(*args, **kwargs)
        self.attr_start, self.attr_min = getattr(self, self.attr)
        setattr(self, 'current_%s' % self.attr, self.attr_start)
        self.assist_flag = True

    def init(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        super(LinearlyDecreasingAttribute, self).init()
        if isinstance(self, SaveHDF):
            self.add_dtype(self.attr, 'f8', (1,))

    def _linear_change(self, start_val, end_val, decay_time):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if start_val == end_val:
            return end_val
        else:
            elapsed_time = self.get_time() - self.task_start_time
            temp = start_val - elapsed_time/decay_time*(start_val-end_val)
            if start_val > end_val:
                return max(temp, end_val)
            elif start_val < end_val:
                return min(temp, end_val)

    def update_level(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        decay_time = float(getattr(self, '%s_time' % self.attr)) #self.assist_level_time
        current_level = self._linear_change(self.attr_start, self.attr_min, decay_time)
        setattr(self, 'current_%s' % self.attr, current_level) 
        if self.assist_flag and getattr(self, 'current_%s' % self.attr) == self.attr_min:
            print "%s at final value after %d successful trials" % (self.attr, self.calc_n_rewards())
            self.assist_flag = False

        if self.cycle_count % (self.fps * sec_per_min) == 0 and self.assist_flag:
            print "%s: " % self.attr, getattr(self, 'current_%s' % self.attr)

    def _cycle(self):
        '''
        Update and save the current attribute value before calling the next _cycle in the MRO
        '''
        self.update_level()
        if hasattr(self, 'task_data'):
            self.task_data[self.attr] = getattr(self, 'current_%s' % self.attr)

        super(LinearlyDecreasingAttribute, self)._cycle()

class LinearlyDecreasingAssist(LinearlyDecreasingAttribute):
    ''' Docstring '''
    assist_level = traits.Tuple((0.0, 0.0), desc="Level of assist to apply to BMI output")
    assist_level_time = traits.Float(600, desc="Number of seconds to go from initial to minimum assist level")    
    attr = 'assist_level'

class LinearlyDecreasingHalfLife(LinearlyDecreasingAttribute):
    ''' Docstring '''
    half_Life = traits.Tuple((450., 450.), desc="Initial and final half life for CLDA")
    half_life_time = traits.Float(600, desc="Number of seconds to go from initial to final half life")
    attr = 'half_life'
