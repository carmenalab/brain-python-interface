''' Tasks which don't include any visuals or bmi, such as laser-only or camera-only tasks'''

from riglib.experiment import LogExperiment, Sequence
from features.laser_features import DigitalWave
from riglib.experiment import traits
from riglib.stereo_opengl.window import Window
from built_in_tasks.target_graphics import VirtualCircularTarget, target_colors
import itertools
import numpy as np


class Conditions(Sequence):

    status = dict(
        wait = dict(start_trial="trial", stop=None),
        trial = dict(end_trial="wait"),
    )
    
    wait_time = traits.Float(5.0, desc="Inter-trial interval (s)")
    trial_time = traits.Float(1.0, desc="Trial length (s)")
    sequence_generators = ['null_sequence']

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause
    
    def _test_end_trial(self, ts):
        return ts > self.trial_time

    @staticmethod
    def gen_random_conditions(nreps, *args, replace=False):
        ''' Generate random sequence of all combinations of the given arguments'''
        unique = list(itertools.product(*args))
        conds = np.random.choice(nreps*len(unique), nreps*len(unique), replace=replace)
        seq = [list(unique[i % len(unique)]) for i in conds]
        return tuple(zip(*seq))

    @staticmethod
    def gen_conditions(nreps, *args, ascend=True):
        ''' Generate a sequential sequence of all combinations of the given arguments'''
        unique = list(itertools.product(*args))
        tiled = np.tile(np.asarray(unique), nreps)
        if ascend:
            seq = tiled.flatten()
        else:
            seq = np.flipud(tiled).flatten()
        return tuple(zip(*seq))

    @staticmethod
    def null_sequence(ntrials):
        return range(ntrials)

class LaserConditions(Conditions):

    trial_time = None
    laser_trigger_pin = traits.Int(10, desc="GPIO pin used to trigger laser")
    sequence_generators = ['pulse', 'square_wave']

    def __init__(self, *args, **kwargs):
        self.laser_thread = None
        self.power = 0.
        self.edges = []
        # gpio must be defined elsewhere
        super().__init__(*args, **kwargs)

    def init(self):
        self.add_dtype('power', 'f8', (1,))
        super().init()

    def _cycle(self):
        self.task_data['power'] = self.power.copy()

    def _parse_next_trial(self):
        self.power, self.edges = self.next_trial

    def _start_trial(self):
        # TODO set laser power
        # Trigger digital wave
        wave = DigitalWave(self.gpio, pin=self.laser_trigger_pin)
        wave.set_edges(self.edges, True)
        wave.start()
        self.laser_thread = wave

    def _start_wait(self):
        # Turn laser off in between trials
        wave = DigitalWave(self.gpio, pin=self.laser_trigger_pin)
        wave.set_edges([0], False)
        wave.start()
        super()._start_wait()
        
    def _test_end_trial(self, ts):
        return (self.laser_thread is None) or (not self.laser_thread.is_alive())

    @staticmethod
    def pulse(nreps=100, duration=[0.005], power=[1], uniformsampling=True, ascending=False):
        '''
        Generates a sequence of laser pulse trains.

        Parameters
        ----------
        nreps : int
            The number of repetitions of each unique condition.
        duration: list of floats
            The duration of each pulse. Can be a list, randomly sampled
        power : list of floats
            Power for each pulse. Can be a list, randomly sampled

        Returns
        -------
        seq : [nreps*len(duration)*len(power) x 2] list of laser powers and edge sequences

        '''
        duration = make_list_of_float(duration)
        power = make_list_of_float(power)
        if uniformsampling:
            dur_seq, pow_seq = Conditions.gen_random_conditions(nreps, duration, power)
        else:
            dur_seq, pow_seq = Conditions.gen_conditions(nreps, duration, power, ascend=ascending)
        edge_seq = map(lambda dur: [0, dur], dur_seq)
        return list(zip(pow_seq, edge_seq))

    @staticmethod
    def square_wave(nreps=100, freq=[20], duration=[0.005], power=[1], uniformsampling=True, ascending=False):
        '''
        Generates a sequence of laser square waves.

        Parameters
        ----------
        nreps : int
            The number of repetitions of each unique condition.
        freq : list of floats
            The frequency for each square wave. Can be a list, randomly sampled
        duration: list of floats
            The duration of each square wave. Can be a list, randomly sampled
        power : list of floats
            Power for each square wave. Can be a list, randomly sampled

        Returns
        -------
        pow_seq : [nreps*len(duration)*len(power)] list of laser powers
        edge_seq : [nreps*len(duration)*len(power) x 2] array of laser wave edge times

        '''
        freq = make_list_of_float(freq)
        duration = make_list_of_float(duration)
        power = make_list_of_float(power)
        if uniformsampling:
            freq_seq, dur_seq, pow_seq = Conditions.gen_random_conditions(nreps, freq, duration, power)
        else:
            freq_seq, dur_seq, pow_seq = Conditions.gen_conditions(nreps, freq, duration, power, ascend=ascending)
        edge_seq = map(lambda freq, dur: DigitalWave.square_wave(freq, dur), freq_seq, dur_seq)
        return list(zip(pow_seq, edge_seq))

class MonkeyTraining(Window):

    status = dict(
        wait = dict(start_trial="trial", stop=None),
        trial = dict(end_trial="wait"),
    )

    state = "wait"
    
    background = (0,0,0,1)
    target_color = traits.OptionsList(tuple(target_colors.keys()), desc="Color of the target")
    target_radius = traits.Float(5, desc="Radius of targets in cm")
    target_location = np.array([0, 0, 0])
    
    wait_time = traits.Float(5.0, desc="Time in between trials (s). If set to 0, then trials begin with keypress")
    trial_time = traits.Float(1.0, desc="Trial length (s)")

    def _test_start_trial(self, ts):
        if self.wait_time == 0:
            from pygame import K_ESCAPE
            return self.event is not None and self.event[0] != K_ESCAPE # keypress
        else:
            return ts > self.wait_time and not self.pause
    
    def _test_end_trial(self, ts):
        return ts > self.trial_time

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = VirtualCircularTarget(target_radius=self.target_radius, target_color=target_colors[self.target_color])
        for model in self.target.graphics_models:
            self.add_model(model)

    def _start_trial(self):
        self.target.show()

    def _start_wait(self):
        self.target.hide()

class WhiteMatterCamera(LogExperiment):
    pass

def make_list_of_float(maybe_a_float):
    try:
        _ = iter(maybe_a_float)
    except TypeError:
        return [maybe_a_float]
    else:
        return maybe_a_float