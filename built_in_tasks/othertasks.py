''' Tasks which don't include any visuals or bmi, such as laser-only or camera-only tasks'''

from riglib.experiment import LogExperiment, Sequence
from features.laser_features import DigitalWave
from riglib.experiment import traits
import itertools
import numpy as np

MAX_RECORD_EDGES = 32

class Conditions(Sequence):

    status = dict(
        wait = dict(start_trial="trial"),
        trial = dict(end_trial="wait", stoppable=False, end_state=True),
    )
    
    wait_time = traits.Float(5.0, desc="Inter-trial interval (s)")
    trial_time = traits.Float(1.0, desc="Trial duration (s)")
    sequence_generators = ['null_sequence']

    def init(self):
        self.trial_dtype = np.dtype([('trial', 'u4'), ('index', 'u4')])
        super().init()

    def _parse_next_trial(self):
        self.trial_index = self.next_trial

        # Send record of trial to sinks
        self.trial_record['trial'] = self.calc_trial_num()
        self.trial_record['index'] = self.trial_index
        self.sinks.send("trials", self.trial_record)

    def _test_start_trial(self, ts):
        return ts > self.wait_time and not self.pause
    
    def _test_end_trial(self, ts):
        return ts > self.trial_time

    def _start_trial(self):
        self.sync_event('TRIAL_START', self.trial_index)

    def _end_trial(self):
        self.sync_event('TRIAL_END')

    @classmethod
    def get_desc(cls, params, log_summary):
        duration = round(log_summary['runtime'] / 60, 1)
        return "{} trials in {} min".format(
            log_summary['n_trials'], duration)

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
        conds = np.tile(range(len(unique)), (nreps, 1)).T.flatten()
        if not ascend: # descending
            conds = np.flipud(conds)
        seq = [[i % len(unique)] + list(unique[i % len(unique)]) for i in conds] # list of [index, arg1, arg2, ..., argn]
        return tuple(zip(*seq))

    @staticmethod
    def null_sequence(ntrials=100):
        return [0 for _ in range(ntrials)]

class LaserConditions(Conditions):

    sequence_generators = ['single_laser_pulse', 'single_laser_square_wave', 'dual_laser_square_wave']
    exclude_parent_traits = ['trial_time']

    def __init__(self, *args, **kwargs):
        self.laser_threads = []
        super().__init__(*args, **kwargs)

    def init(self):
        self.trial_dtype = np.dtype([
            ('trial', 'u4'), 
            ('index', 'u4'),
            ('laser', 'S32'),
            ('power', 'f8'),
            ('edges', 'f8', MAX_RECORD_EDGES),
            ])
        super(Conditions, self).init()
    
    def run(self):
        if not (hasattr(self, 'lasers') and len(self.lasers) > 0):
            import io
            self.terminated_in_error = True
            self.termination_err = io.StringIO()
            self.termination_err.write("No laser feature enabled, cannot init LaserConditions")
            self.termination_err.seek(0)
            self.state = None
        super().run() 

    def _parse_next_trial(self):
        self.trial_index, self.laser_powers, self.laser_edges = self.next_trial
        if len(self.laser_powers) < len(self.lasers) or len(self.laser_edges) < len(self.lasers):
            raise AttributeError("Not enough laser sequences for the number of lasers enabled")

        # Send record of trial to sinks
        self.trial_record['trial'] = self.calc_trial_num()
        self.trial_record['index'] = self.trial_index
        for idx in range(len(self.lasers)):
            self.trial_record['laser'] = self.lasers[idx].name
            self.trial_record['power'] = self.laser_powers[idx]
            record_laser_edges = self.laser_edges[idx][:MAX_RECORD_EDGES]
            self.trial_record['edges'] = np.pad(record_laser_edges, (0, MAX_RECORD_EDGES - len(record_laser_edges)), constant_values=np.nan)
            self.sinks.send("trials", self.trial_record)

    def _start_trial(self):
        super()._start_trial()
        for idx in range(len(self.lasers)):
            laser = self.lasers[idx]
            edges = self.laser_edges[idx]
            # set laser power
            power = self.laser_powers[idx]
            laser.set_power(power)
            # Trigger digital wave
            wave = DigitalWave(laser, mask=1<<laser.port)
            wave.set_edges(edges, True)
            wave.start()
            self.laser_threads.append(wave)

    def _end_trial(self):
        super()._end_trial()
        # Turn laser off in between trials in case it ended on a rising edge
        for idx in range(len(self.lasers)):
            laser = self.lasers[idx]
            wave = DigitalWave(laser, mask=1>>laser.port)
            wave.set_edges([0], False)
            wave.start()
        
    def _test_end_trial(self, ts):
        return all([not t.is_alive() for t in self.laser_threads])

    @staticmethod
    def single_laser_pulse(nreps=100, duration=[0.005], power=[1], uniformsampling=True, ascending=False):
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
        return list(zip(idx, [[p] for p in pow_seq], [[e] for e in edge_seq]))

    @staticmethod
    def single_laser_square_wave(nreps=100, freq=[20], duration=[0.5], power=[1], uniformsampling=True, ascending=False):
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
        return list(zip(idx, [[p] for p in pow_seq], [[e] for e in edge_seq]))

    @staticmethod
    def dual_laser_square_wave(nreps=100, freq_1=[20], freq_2=[20], dur_1=[0.5], dur_2=[0.5], duty_cycle_1=[0.5], duty_cycle_2=[0.5], 
        phase_delay_1=[0], phase_delay_2=[0], power_1=[1], power_2=[1], uniformsampling=True, ascending=False):
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
        freq_1 = make_list_of_float(freq_1)
        freq_2 = make_list_of_float(freq_2)
        dur_1 = make_list_of_float(dur_1)
        dur_2 = make_list_of_float(dur_2)
        duty_cycle_1 = make_list_of_float(duty_cycle_1)
        duty_cycle_2 = make_list_of_float(duty_cycle_2)
        phase_delay_1 = make_list_of_float(phase_delay_1)
        phase_delay_2 = make_list_of_float(phase_delay_2)
        power_1 = make_list_of_float(power_1)
        power_2 = make_list_of_float(power_2)
        if uniformsampling:
            idx, freq_1_seq, freq_2_seq, dur_1_seq, dur_2_seq, duty_cycle_1_seq, duty_cycle_2_seq, phase_delay_1_seq, phase_delay_2_seq, power_1_seq, power_2_seq = \
            Conditions.gen_random_conditions(nreps, freq_1, freq_2, dur_1, dur_2, duty_cycle_1, duty_cycle_2, phase_delay_1, phase_delay_2, power_1, power_2)
        else:
            idx, freq_1_seq, freq_2_seq, dur_1_seq, dur_2_seq, duty_cycle_1_seq, duty_cycle_2_seq, phase_delay_1_seq, phase_delay_2_seq, power_1_seq, power_2_seq = \
            Conditions.gen_conditions(nreps, freq_1, freq_2, dur_1, dur_2, duty_cycle_1, duty_cycle_2, phase_delay_1, phase_delay_2, power_1, power_2, ascend=ascending)
        edge_1_seq= map(lambda freq, dur, dc, delay: DigitalWave.square_wave(freq, dur, duty_cycle=dc, phase_delay=delay), freq_1_seq, dur_1_seq, duty_cycle_1_seq, phase_delay_1_seq)
        edge_2_seq = map(lambda freq, dur, dc, delay: DigitalWave.square_wave(freq, dur, duty_cycle=dc, phase_delay=delay), freq_2_seq, dur_2_seq, duty_cycle_2_seq, phase_delay_2_seq)
        return list(zip(idx, [[p1, p2] for p1, p2 in zip(power_1_seq, power_2_seq)], 
                             [[e1, e2] for e1, e2 in zip(edge_1_seq, edge_2_seq)]))


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