''' Mock experiment classes for testing '''
import unittest
import time

from fsm import FSM, ThreadedFSM, FSMTable, StateTransitions


event1to2 = [False, True,  False, False, False, False, False, False]
event1to3 = [False, False, False, False, True,  False, False, False]
event2to3 = [False, False, True,  False, False, False, False, False]
event2to1 = [False, False, False, False, False, False, False, False]
event3to2 = [False, False, False, False, False, True,  False, False]
event3to1 = [False, False, False, True,  False, False, False, False]

class MockFSM(ThreadedFSM):
    status = FSMTable(
        state1=StateTransitions(event1to2='state2', event1to3='state3'),
        state2=StateTransitions(event2to3='state3', event2to1='state1'),
        state3=StateTransitions(event3to2='state2', event3to1='state1'),
    )
    state = 'state1'

    def __init__(self, *args, **kwargs):
        self.iter_idx = 0
        super(MockFSM, self).__init__(*args, **kwargs)

    def _cycle(self):
        self.iter_idx += 1
        super(MockFSM, self)._cycle()

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


class TestFSM(unittest.TestCase):
    def setUp(self):
        self.exp = MockFSM()

    def test_exp_fsm_output(self):
        """Experiment state sequence follows reference sequence"""
        self.exp.run_sync()
        self.assertEqual(self.exp.event_log,
            [('state1', 'event1to2', 1), ('state2', 'event2to3', 2), ('state3', 'event3to1', 3), 
             ('state1', 'event1to3', 4), ('state3', 'event3to2', 5), ('state2', 'stop', 7)])

class ThreadedFSM(unittest.TestCase):
    def setUp(self):
        self.exp = MockFSM()
            
    def test_thread_start(self):
        """Experiment FSM execution should run in a new thread and return when state sequence is complete."""
        self.exp.start()
        self.exp.join()
        self.assertEqual(self.exp.event_log,
            [('state1', 'event1to2', 1), ('state2', 'event2to3', 2), ('state3', 'event3to1', 3), 
             ('state1', 'event1to3', 4), ('state3', 'event3to2', 5), ('state2', 'stop', 7)])

        self.assertEqual(self.exp.cycle_count, 7)

    # def test_thread_stop(self):
    #     """Experiment execution in remote thread should terminate when 'end_task' is called"""
    #     # Experiment should never start if the initial state is None
    #     exp = ThreadedFSM()
    #     exp.state = None
    #     exp.start()
    #     exp.join()
    #     self.assertEqual(exp.cycle_count, 0)

    #     exp = ThreadedFSM()
    #     exp.state = "wait"
    #     exp.start()
    #     time.sleep(1)
    #     exp.end_task()

    #     # number of cycles should be about 60. Wide margin so that test always passes
    #     # (not exact due to threading and imprecise timing)
    #     margin = 20
    #     self.assertTrue(exp.cycle_count > exp.fps - margin)
    #     self.assertTrue(exp.cycle_count < exp.fps + margin)        

if __name__ == '__main__':
    unittest.main()