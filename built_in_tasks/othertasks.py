''' Tasks which don't include any visuals or bmi, such as laser-only or camera-only tasks'''

from riglib.experiment import LogExperiment, Sequence
from features.laser_features import DigitalWave
from riglib.experiment import traits
import itertools
import numpy as np


class Conditions(Sequence):

    status = dict(
        wait = dict(start_trial="trial", stop=None),
        trial = dict(end_trial="wait"),
    )
    
    wait_time = traits.Float(5.0, desc="Inter-trial interval (s)")
    trial_time = traits.Float(1.0, desc="Trial duration (s)")
    sequence_generators = ['null_sequence']

    def _parse_next_trial(self):
        self.trial_index = self.next_trial

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause
    
    def _test_end_trial(self, ts):
        return ts > self.trial_time

    @staticmethod
    def gen_random_conditions(nreps, *args, replace=False):
        ''' Generate random sequence of all combinations of the given arguments'''
        unique = list(itertools.product(*args))
        conds = np.random.choice(nreps*len(unique), nreps*len(unique), replace=replace)
        seq = [[i % len(unique)] + list(unique[i % len(unique)]) for i in conds] # list of [index, arg1, arg2, ..., argn]
        return tuple(zip(*seq))

    @staticmethod
    def gen_conditions(nreps, *args, ascend=True):
        ''' Generate a sequential sequence of all combinations of the given arguments'''
        unique = list(itertools.product(*args))
        conds = np.tile(range(len(unique)), nreps)
        if not ascend: # descending
            conds = np.flipud(conds)
        seq = [[i % len(unique)] + list(unique[i % len(unique)]) for i in conds] # list of [index, arg1, arg2, ..., argn]
        return tuple(zip(*seq))

    @staticmethod
    def null_sequence(ntrials):
        return range(ntrials)

class LaserConditions(Conditions):

    laser_trigger_pin = traits.Int(10, desc="GPIO pin used to trigger laser")
    sequence_generators = ['pulse', 'square_wave']
    exclude_parent_traits = Conditions.exclude_parent_traits + ['trial_time']

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
        self.trial_index, self.power, self.edges = self.next_trial

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
        seq : (nreps*len(duration)*len(power) x 3) tuple of trial indices, laser powers, and edge sequences

        '''
        duration = make_list_of_float(duration)
        power = make_list_of_float(power)
        if uniformsampling:
            idx, dur_seq, pow_seq = Conditions.gen_random_conditions(nreps, duration, power)
        else:
            idx, dur_seq, pow_seq = Conditions.gen_conditions(nreps, duration, power, ascend=ascending)
        edge_seq = map(lambda dur: [0, dur], dur_seq)
        return list(zip(idx, pow_seq, edge_seq))

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
        seq : (nreps*len(duration)*len(power)*len(freq) x 3) tuple of trial indices, laser powers, and edge sequences

        '''
        freq = make_list_of_float(freq)
        duration = make_list_of_float(duration)
        power = make_list_of_float(power)
        if uniformsampling:
            idx, freq_seq, dur_seq, pow_seq = Conditions.gen_random_conditions(nreps, freq, duration, power)
        else:
            idx, freq_seq, dur_seq, pow_seq = Conditions.gen_conditions(nreps, freq, duration, power, ascend=ascending)
        edge_seq = map(lambda freq, dur: DigitalWave.square_wave(freq, dur), freq_seq, dur_seq)
        return list(zip(idx, pow_seq, edge_seq))


####################
# Helper functions #
####################
def make_list_of_float(maybe_a_float):
    try:
        _ = iter(maybe_a_float)
    except TypeError:
        return [maybe_a_float]
    else:
        return maybe_a_float