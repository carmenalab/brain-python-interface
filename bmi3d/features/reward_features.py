'''
Reward delivery features
'''
import time
import os
import subprocess
from ..riglib.experiment import traits
from ..riglib.audio import AudioPlayer
import serial, glob

###### CONSTANTS
sec_per_min = 60

class RewardSystem(traits.HasTraits):
    '''
    Feature for the current reward system in Amy Orsborn Lab
    '''
    trials_per_reward = traits.Float(1, desc='Number of successful trials before solenoid is opened')

    def __init__(self, *args, **kwargs):
        from ..riglib import reward
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
        if hasattr(super(), '_test_reward_end'):
            super()._test_reward_end()

    def _end_reward(self):
        self.reward.off()
        if hasattr(super(), '_end_reward'):
            super()._end_reward()

audio_path = os.path.join(os.path.dirname(__file__), '../riglib/audio')

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

""""" BELOW THIS IS ALL THE OLD CODE ASSOCIATED WITH REWARD FEATURES"""


class RewardSystem_Crist(traits.HasTraits):
    '''
    Feature for the Crist solenoid reward system
    '''
    trials_per_reward = traits.Float(1, desc='Number of successful trials before solenoid is opened')

    def __init__(self, *args, **kwargs):
        from ..riglib import reward_crist
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


