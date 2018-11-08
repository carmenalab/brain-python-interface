from riglib import experiment
reload(experiment)
from riglib.experiment import LogExperiment, FSMTable, StateTransitions


event1to2 = [False, True,  False, False, False, False, False, False]
event1to3 = [False, False, False, False, True,  False, False, False]
event2to3 = [False, False, True,  False, False, False, False, False]
event2to1 = [False, False, False, False, False, False, False, False]
event3to2 = [False, False, False, False, False, True,  False, False]
event3to1 = [False, False, False, True,  False, False, False, False]

class BaseExp(LogExperiment):
    status = FSMTable(
        state1=StateTransitions(event1to2='state2', event1to3='state3'),
        state2=StateTransitions(event2to3='state3', event2to1='state1'),
        state3=StateTransitions(event3to2='state2', event3to1='state1'),
    )
    state = 'state1'

    def __init__(self, *args, **kwargs):
        self.iter_idx = 0
        super(BaseExp, self).__init__(*args, **kwargs)

    def _cycle(self):
        self.iter_idx += 1
        super(BaseExp, self)._cycle()

    def _start_state3(self): print("start state3")
    def _while_state3(self): print("while state3")
    def _end_state3(self): print("end state3")
    def _start_state2(self): print("start state2")
    def _while_state2(self): print("while state2")
    def _end_state2(self): print("end state2")
    def _start_state1(self): print("start state1")
    def _while_state1(self): print("while state1")
    def _end_state1(self): print("end state1")
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


exp = BaseExp()
exp.run_sync()

if exp.event_log == [('state1', 'event1to2', 1), ('state2', 'event2to3', 2), ('state3', 'event3to1', 3), ('state1', 'event1to3', 4), ('state3', 'event3to2', 5), ('state2', 'stop', 7)]:
    print("event transitions match!")



## The output should look like 
# start state1
# while state1
# end state1
# start state2
# while state2
# end state2
# start state3
# while state3
# end state3
# start state1
# while state1
# end state1
# start state3
# while state3
# end state3
# start state2
# while state2
# while state2
# end state2
