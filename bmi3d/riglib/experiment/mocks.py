''' Mock experiment classes for testing '''
from .experiment import LogExperiment, FSMTable, StateTransitions, Sequence

event1to2 = [False, True,  False, False, False, False, False, False]
event1to3 = [False, False, False, False, True,  False, False, False]
event2to3 = [False, False, True,  False, False, False, False, False]
event2to1 = [False, False, False, False, False, False, False, False]
event3to2 = [False, False, False, False, False, True,  False, False]
event3to1 = [False, False, False, True,  False, False, False, False]

class MockLogExperiment(LogExperiment):
    status = FSMTable(
        state1=StateTransitions(event1to2='state2', event1to3='state3'),
        state2=StateTransitions(event2to3='state3', event2to1='state1'),
        state3=StateTransitions(event3to2='state2', event3to1='state1'),
    )
    state = 'state1'

    def __init__(self, *args, **kwargs):
        self.iter_idx = 0
        super(MockLogExperiment, self).__init__(*args, **kwargs)

    def _cycle(self):
        self.iter_idx += 1
        super(MockLogExperiment, self)._cycle()

    def _start_state3(self): pass
    def _while_state3(self): pass
    def _end_state3(self): pass
    def _start_state2(self): pass
    def _while_state2(self): pass
    def _end_state2(self): pass
    def _start_state1(self): pass
    def _while_state1(self): pass
    def _end_state1(self): pass
    ################## State trnasition test functions ##################
    def _test_event3to1(self, time_in_state): return event3to1[self.iter_idx]
    def _test_event3to2(self, time_in_state): return event3to2[self.iter_idx]
    def _test_event2to3(self, time_in_state): return event2to3[self.iter_idx]
    def _test_event2to1(self, time_in_state): return event2to1[self.iter_idx]
    def _test_event1to3(self, time_in_state): return event1to3[self.iter_idx]
    def _test_event1to2(self, time_in_state): return event1to2[self.iter_idx]
    def _test_stop(self, time_in_state):
        return self.iter_idx >= len(event1to2) - 1

    def get_time(self):
        return self.iter_idx



class MockSequence(Sequence):
    event1to2 = [False, True,  False, True, False, True, False, True, False, True, False]
    event1to3 = [False, False, False, False, False,  False, False, False, False, False, False]
    event2to3 = [False, False, False,  False, False, False, False, False, False, False, False]
    event2to1 = [False, False, True, False, True, False, True, False, True, False, False]
    event3to2 = [False, False, False, False, False, False,  False, False, False, False, False]
    event3to1 = [False, False, False, False,  False, False, False, False, False, False, False]

    status = FSMTable(
        wait=StateTransitions(event1to2='state2', event1to3='state3'),
        state2=StateTransitions(event2to3='state3', event2to1='wait'),
        state3=StateTransitions(event3to2='state2', event3to1='wait'),
    )
    state = 'wait'

    def __init__(self, *args, **kwargs):
        self.iter_idx = 0
        self.target_history = []
        super(MockSequence, self).__init__(*args, **kwargs)

    def _cycle(self):
        self.iter_idx += 1
        super(MockSequence, self)._cycle()

    def _start_state3(self): pass
    def _while_state3(self): pass
    def _end_state3(self): pass
    def _start_state2(self): pass
    def _while_state2(self): pass
    def _end_state2(self): pass
    def _start_state1(self): pass
    def _while_state1(self): pass
    def _end_state1(self): pass
    ################## State trnasition test functions ##################
    def _test_event3to1(self, time_in_state): return self.event3to1[self.iter_idx]
    def _test_event3to2(self, time_in_state): return self.event3to2[self.iter_idx]
    def _test_event2to3(self, time_in_state): return self.event2to3[self.iter_idx]
    def _test_event2to1(self, time_in_state): return self.event2to1[self.iter_idx]
    def _test_event1to3(self, time_in_state): return self.event1to3[self.iter_idx]
    def _test_event1to2(self, time_in_state): return self.event1to2[self.iter_idx]
    def _test_stop(self, time_in_state):
        return self.iter_idx >= len(event1to2) - 1

    def get_time(self):
        return self.iter_idx

    def _start_wait(self):
        super(MockSequence, self)._start_wait()
        self.target_history.append(self.next_trial)


class MockSequenceWithGenerators(Sequence):
    status = FSMTable(
        wait=StateTransitions(start_trial="trial"),
        trial=StateTransitions(target_reached="reward"),
        reward=StateTransitions(reward_complete="wait"),
    )
    state = 'wait'

    sequence_generators = ['gen_fn1', 'gen_fn2']

    def __init__(self, *args, **kwargs):
        self.target_history = []
        self.sim_state_seq = []
        for k in range(4):
            self.sim_state_seq += [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2]

        self.current_state = None
        super().__init__(*args, **kwargs)

    def init(self):
        self.add_dtype("target_state", int, (1,))
        self.add_dtype("current_state", int, (1,))
        super().init()

    def get_time(self):
        return self.cycle_count * 1.0/60

    @staticmethod
    def gen_fn1(n_targets=4):
        target_seq = [1, 2] * n_targets
        return [{"target":x} for x in target_seq]

    @staticmethod
    def gen_fn2(n_targets=4):
        target_seq = [1, 2] * n_targets
        return [{"target":x} for x in target_seq]

    def _test_start_trial(self, ts):
        return True

    def _test_reward_complete(self, ts):
        return True

    def _cycle(self):
        self.current_state = self.sim_state_seq[self.cycle_count % len(self.sim_state_seq)]
        self.task_data["target_state"] = self._gen_target
        self.task_data["current_state"] = self.current_state

        if self.cycle_count == 21:
            self.record_annotation("test annotation")

        super()._cycle()

    def _test_target_reached(self, ts):
        return self.current_state == self._gen_target

from . import traits
class MockSequenceWithTraits(MockSequenceWithGenerators):
    options_trait = traits.OptionsList(["option1", "option2"], desc='options', label="Options")
    float_trait = traits.Float(15, desc="float", label="Float")
