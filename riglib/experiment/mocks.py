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
        wait=StateTransitions(event1to2='state2', event1to3='state3'),
        state2=StateTransitions(event2to3='state3', event2to1='wait'),
        state3=StateTransitions(event3to2='state2', event3to1='wait'),
    )
    state = 'wait'

    sequence_generators = ['gen_fn1', 'gen_fn2']

    def __init__(self, *args, **kwargs):
        self.iter_idx = 0
        self.target_history = []
        super(MockSequence, self).__init__(*args, **kwargs)

    def _cycle(self):
        self.iter_idx += 1
        super(MockSequence, self)._cycle()

    @staticmethod
    def gen_fn1():
    	return [1, 2, 1, 2, 1, 2, 1, 2]

    @staticmethod
    def gen_fn2():
    	return [3, 4, 3, 4, 3, 4, 3, 4]

