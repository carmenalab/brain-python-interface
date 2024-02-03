'''
Reward delivery features
'''
import time
import os
import subprocess
from riglib.experiment import traits
from riglib.experiment.experiment import control_decorator
from riglib.audio import AudioPlayer
from built_in_tasks.target_graphics import VirtualRectangularTarget
import numpy as np
import serial, glob

###### CONSTANTS
sec_per_min = 60

class RewardSystem(traits.HasTraits):
    '''
    Feature for the current reward system in Amy Orsborn Lab
    '''
    trials_per_reward = traits.Float(1, desc='Number of successful trials before solenoid is opened')

    def __init__(self, *args, **kwargs):
        from riglib import reward
        super().__init__(*args, **kwargs)
        self.reward = reward.open()
        self.reportstats['Reward #'] = 0

    def run(self):
        if self.reward is None:
            raise Exception('Reward system could not be activated')
        super().run()

    def _start_reward(self):
        if hasattr(super(), '_start_reward'):
            super()._start_reward()
        self.reportstats['Reward #'] += 1
        if self.reportstats['Reward #'] % self.trials_per_reward == 0:
            self.reward.on()

    def _test_reward_end(self, ts):
        if self.reportstats['Reward #'] % self.trials_per_reward == 0:
            return ts > self.reward_time
        else:
            return True

    def _end_reward(self):
        self.reward.off()
        if hasattr(super(), '_end_reward'):
            super()._end_reward()

    @control_decorator
    def manual_reward(duration=0.5, static=True):
        from riglib import reward
        reward_sys = reward.open()
        float_dur = float(duration)  # these parameters always end up being strings
        reward_sys.async_drain(float_dur)

audio_path = os.path.join(os.path.dirname(__file__), '../riglib/audio')

class PelletReward(RewardSystem):
    '''
    Trigger pellet rewards.    
    '''
    pellets_per_reward = traits.Int(1, desc='The number of pellets to dispense per reward.')      

    def __init__(self, *args, **kwargs):
        from riglib.tablet_reward import RemoteReward
        super(RewardSystem, self).__init__(*args, **kwargs)
        self.reward = RemoteReward()
        self.reportstats['Reward #'] = 0

    def _start_reward(self):
        if hasattr(super(RewardSystem, self), '_start_reward'):
            super(RewardSystem, self)._start_reward()
        self.reportstats['Reward #'] += 1
        
        if self.reportstats['Reward #'] % self.trials_per_reward == 0:
            for _ in range(self.pellets_per_reward): # call trigger num of pellets_per_reward time
                self.reward.trigger()
                time.sleep(0.5) # wait for 0.5 seconds

    def _end_reward(self):
        if hasattr(super(RewardSystem, self), '_end_reward'):
            super(RewardSystem, self)._end_reward()

    @control_decorator
    def manual_reward( static=True):
        from riglib.tablet_reward import RemoteReward
        reward_sys = RemoteReward()
        reward_sys.reward.trigger()


class RewardAudio(traits.HasTraits):
    '''
    Play a sound in any reward state. Need to add other reward states you want to be included.
    '''

    files = [f for f in os.listdir(audio_path) if '.wav' in f]
    reward_sound = traits.OptionsList(files, desc="File in riglib/audio to play on each reward")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_player = AudioPlayer(self.reward_sound)

    def _start_reward(self):
        if hasattr(super(), '_start_reward'):
            super()._start_reward()
        self.reward_player.play()

class PenaltyAudio(traits.HasTraits):
    '''
    Play a sound in any penalty state. Have to define a new _start method for each different
    penalty state that might occur.
    '''
    files = list(reversed([f for f in os.listdir(audio_path) if '.wav' in f]))
    penalty_sound = traits.OptionsList(files, desc="File in riglib/audio to play on each penalty")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty_player = AudioPlayer(self.penalty_sound)

    def _start_hold_penalty(self):
        if hasattr(super(), '_start_hold_penalty'):
            super()._start_hold_penalty()
        self.penalty_player.play()
    
    def _start_delay_penalty(self):
        if hasattr(super(), '_start_delay_penalty'):
            super()._start_delay_penalty()
        self.penalty_player.play()

    def _start_reach_penalty(self):
        if hasattr(super(), '_start_reach_penalty'):
            super()._start_reach_penalty()
        self.penalty_player.play()
    
    def _start_timeout_penalty(self):
        if hasattr(super(), '_start_timeout_penalty'):
            super()._start_timeout_penalty()
        self.penalty_player.play()

class PenaltyAudioMulti(traits.HasTraits):
    '''
    Separate penalty sounds for each type of penalty.
    '''
    
    hold_penalty_sound = "incorrect.wav"
    delay_penalty_sound = "buzzer.wav"
    timeout_penalty_sound = "incorrect.wav"
    reach_penalty_sound = "incorrect.wav"
    tracking_out_penalty_sound = "buzzer.wav"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold_penalty_player = AudioPlayer(self.hold_penalty_sound)
        self.delay_penalty_player = AudioPlayer(self.delay_penalty_sound)
        self.timeout_penalty_player = AudioPlayer(self.timeout_penalty_sound)
        self.reach_penalty_player = AudioPlayer(self.reach_penalty_sound)
        self.tracking_out_penalty_player = AudioPlayer(self.tracking_out_penalty_sound)

    def _start_hold_penalty(self):
        if hasattr(super(), '_start_hold_penalty'):
            super()._start_hold_penalty()
        self.hold_penalty_player.play()
    
    def _start_delay_penalty(self):
        if hasattr(super(), '_start_delay_penalty'):
            super()._start_delay_penalty()
        self.delay_penalty_player.play()
    
    def _start_timeout_penalty(self):
        if hasattr(super(), '_start_timeout_penalty'):
            super()._start_timeout_penalty()
        self.timeout_penalty_player.play()

    def _start_reach_penalty(self):
        if hasattr(super(), '_start_reach_penalty'):
            super()._start_reach_penalty()
        self.reach_penalty_player.play()

    def _start_tracking_out_penalty(self):
        if hasattr(super(), '_start_tracking_out_penalty'):
            super()._start_tracking_out_penalty()
        self.tracking_out_penalty_player.play()

class HoldCompleteRewards(traits.HasTraits):
    '''
    Trigger an extra reward (duration set by hold_reward_time) after successful holds
    '''

    hold_reward_time = traits.Float(0.05)

    def _start_targ_transition(self):
        super()._start_targ_transition()
        if self.target_index + 1 < self.chain_length:

            # We just finished a hold/delay and there are more targets
            self.reward.async_drain(self.hold_reward_time)

class JackpotRewards(traits.HasTraits):
    '''
    Every trials_for_jackpot trials, double reward is administered
    '''

    trials_for_jackpot = traits.Int(5, desc="How many successful trials before a jackpot is delivered")

    def _test_reward_end(self, ts):
        if self.reportstats['Reward #'] % self.trials_for_jackpot == 0:
            return ts > 2*self.reward_time
        elif self.reportstats['Reward #'] % self.trials_per_reward == 0:
            return ts > self.reward_time
        else:
            return True


class ProgressBar(traits.HasTraits):
    '''
    Adds a graphical progress bar for the tracking task which fills up when the cursor is
    inside the target. Does not decrease. When the trial is over, the amount the bar filled
    up scales the amount of reward. Maximum reward is the 'reward_time' parameter.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _while_tracking_in(self):
        super()._while_tracking_in()
        
        # Update progress bar
        self.tracking_frame_index += 1
        self.tracking_rate = self.tracking_frame_index/np.shape(self.targs)[0]*self.bar_width

        if hasattr(self, 'bar'):
            for model in self.bar.graphics_models:
                self.remove_model(model)
            del self.bar

        self.bar = VirtualRectangularTarget(target_width=1.3, target_height=self.tracking_rate, target_color=(0., 1., 0., 0.75), starting_pos=[self.tracking_rate-self.bar_width,0,9])
        for model in self.bar.graphics_models:
            self.add_model(model)
        self.bar.show()

    def _while_reward(self):
        super()._while_reward()

        if hasattr(self, 'bar'):
            for model in self.bar.graphics_models:
                self.remove_model(model)
            del self.bar

        self.reward_frame_index += 1
        reward_numframe = self.reward_time*self.fps
        reward_amount = self.tracking_rate - self.reward_frame_index*self.tracking_rate/reward_numframe
        self.bar = VirtualRectangularTarget(target_width=1.3, target_height=reward_amount, target_color=(0., 1., 0., 0.75), starting_pos=[reward_amount-self.bar_width,0,9])
        for model in self.bar.graphics_models:
            self.add_model(model)
        self.bar.show()        


class TrackingRewards(traits.HasTraits):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def cleanup(self, database, saveid, **kwargs):
            self.reward.off()
            super().cleanup(database, saveid, **kwargs)

        def _start_tracking_in(self):
            super()._start_tracking_in()
            self.trigger_reward = False
            self.reward.off()
            self.reward_start_frame = self.frame_index + self.tracking_reward_interval*self.fps # frame to start first reward
            self.reward_stop_frame = self.reward_start_frame + self.tracking_reward_time*self.fps # frame to stop first reward
            # print('START TRACKING', self.reward_start_frame, self.reward_stop_frame)

        def _while_tracking_in(self):
            super()._while_tracking_in()
            # Give reward for tracking in
            if self.frame_index >= self.reward_start_frame and self.trigger_reward==False:
                self.trigger_reward = True
                self.reward_stop_frame = self.frame_index + self.tracking_reward_time*self.fps # frame to stop current reward
                self.reward_start_frame = self.frame_index + self.tracking_reward_interval*self.fps # frame to start next reward
                self.reward.on()
                # print('REWARD ON', self.frame_index/self.fps)
            if self.frame_index >= self.reward_stop_frame and self.trigger_reward==True:        
                self.trigger_reward = False
                self.reward.off()
                # print('REWARD OFF', self.frame_index/self.fps)

        def _start_tracking_out(self):
            super()._start_tracking_out()
            self.reward.off()

""""" BELOW THIS IS ALL THE OLD CODE ASSOCIATED WITH REWARD FEATURES"""


class RewardSystem_Crist(traits.HasTraits):
    '''
    Feature for the Crist solenoid reward system
    '''
    trials_per_reward = traits.Float(1, desc='Number of successful trials before solenoid is opened')

    def __init__(self, *args, **kwargs):
        from riglib import reward_crist
        super(RewardSystem, self).__init__(*args, **kwargs)
        self.reward = reward_crist.open()
        self.reportstats['Reward #'] = 0

    def _start_reward(self):
        self.reward_start = self.get_time()
        if self.reward is not None:
            self.reportstats['Reward #'] += 1
            if self.reportstats['Reward #'] % self.trials_per_reward == 0:
                self.reward.reward(self.reward_time*1000.)
        super(RewardSystem, self)._start_reward()

    def _test_reward_end(self, ts):
        if self.reportstats['Reward #'] % self.trials_per_reward == 0:
            return ts > self.reward_time
        else:
            return True

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
        self.reportstats['Reward #'] = 0

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

class TTLReward_arduino(TTLReward):
    ''' Same idea as TTL reward, using an arduino instead of nidaq (pin 8)'''
    def __init__(self, *args, **kwargs):
        self.baudrate_rew = 115200
        import serial
        self.port = serial.Serial('/dev/arduino_neurosync', baudrate=self.baudrate_rew)
        super(TTLReward_arduino, self).__init__(*args, **kwargs)
        self.reportstats['Reward #'] = 0

    def _start_reward(self):
        #port = serial.Serial('/dev/arduino_rew', baudrate=self.baudrate_rew)
        self.port.write("a")
        self.reportstats['Reward #'] = self.reportstats['Reward #'] + 1
        self.reward_start = self.get_time() - self.start_time
        super(TTLReward_arduino, self)._start_reward()
        
    def _end_reward(self):
        self.port.write("b")

class TTLReward_arduino_tdt(traits.HasTraits):
    ''' Same idea as TTL reward, using an arduino instead of nidaq (pin 8)'''
    def __init__(self, *args, **kwargs):
        self.baudrate_rew = 115200
        import serial
        #self.port = serial.Serial('/dev/ttyACM0', baudrate=self.baudrate_rew)
        self.port = serial.Serial('/dev/arduino_neurosync', baudrate=self.baudrate_rew)
        
        super(TTLReward_arduino_tdt, self).__init__(*args, **kwargs)
        self.reportstats['Reward #'] = 0

    def _start_reward(self):
        #port = serial.Serial('/dev/arduino_rew', baudrate=self.baudrate_rew)
        self.port.write("j")
        self.reportstats['Reward #'] = self.reportstats['Reward #'] + 1
        self.reward_start = self.get_time() - self.start_time
        super(TTLReward_arduino_tdt, self)._start_reward()

    def _test_reward_end(self, ts):
        return (ts - self.reward_start) > self.reward_time
        
    def _end_reward(self):
        self.port.write("n")



class JuiceLogging(traits.HasTraits):
    '''
    Save screenshots of the juice camera and link them to the task entry that has been created
    '''
    def cleanup(self, database, saveid, **kwargs):
        '''
        See riglib.experiment.Experiment.cleanup for docs on task cleanup.
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
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if dbname == 'default':
            database.save_data(fname, 'juice_log', saveid)
        else:
            database.save_data(fname, 'juice_log', saveid, dbname=dbname)


class ArduinoReward(traits.HasTraits):
    '''During the reward phase, send a timed TTL pulse via the Arduino microcontroller to the reward system'''
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
        self.port = serial.Serial(glob.glob("/dev/ttyACM*")[0], baudrate=115200)
        #self.port.write('n')
        super(ArduinoReward, self).__init__(*args, **kwargs)
        self.reportstats['Reward #'] = 0

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
        super(ArduinoReward, self)._start_reward()
        self.reportstats['Reward #'] = self.reportstats['Reward #'] + 1
        self.port.write('a')
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
        self.port.write('b')


