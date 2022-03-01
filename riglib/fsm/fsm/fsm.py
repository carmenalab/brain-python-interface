"""Finite state machine implementation """
import time
import threading
import traceback
import io

from collections import OrderedDict

min_per_hour = 60.
sec_per_min = 60.


class FSMTable(object):
    def __init__(self, **kwargs):
        self.states = OrderedDict()
        for state_name, transitions in list(kwargs.items()):
            self.states[state_name] = transitions

    def __getitem__(self, key):
        return self.states[key]

    def get_possible_state_transitions(self, current_state):
        return list(self.states[current_state].items())

    def _lookup_next_state(self, current_state, transition_event):
        return self.states[current_state][transition_event]

    def __iter__(self):
        return list(self.states.keys()).__iter__()

    @property
    def trial_end_states(self):
        return [state for state, transitions in self.states.items() if transitions.end_state]

    @staticmethod
    def construct_from_dict(status):
        outward_transitions = OrderedDict()
        for state in status:
            outward_transitions[state] = StateTransitions(**status[state])
        return FSMTable(**outward_transitions)


class StateTransitions(object):
    def __init__(self, stoppable=True, end_state=False, **kwargs):
        self.state_transitions = OrderedDict()
        for event, next_state in list(kwargs.items()):
            self.state_transitions[event] = next_state

        if stoppable and not ('stop' in self.state_transitions):
            self.state_transitions['stop'] = None
        self.end_state = end_state

    def __getitem__(self, key):
        return self.state_transitions[key]

    def __iter__(self):
        transition_events = list(self.state_transitions.keys())
        return transition_events.__iter__()

    def items(self):
        return list(self.state_transitions.items())


class Clock(object):
    def tick(self, fps):
        import time
        time.sleep(1.0/fps)
    def get_time(self):
        return 0

class FSM(object):
    status = FSMTable(
        wait = StateTransitions(start_trial="trial", premature="penalty", stop=None),
        trial = StateTransitions(correct="reward", incorrect="penalty", timeout="penalty"),
        reward = StateTransitions(post_reward="wait"),
        penalty = StateTransitions(post_penalty="wait"),
    )
    state = "wait"
    debug = False
    fps = 60 # frames per second

    log_exclude = set()  # List out state/trigger pairs to exclude from logging

    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.pop('verbose', False)

        # state and event transitions
        self.state_log = []
        self.event_log = []

        self.clock = Clock()

        # Timestamp for rough loop timing
        self.last_time = self.get_time()
        self.cycle_count = 0        

    @property 
    def update_rate(self):
        '''
        Attribute for update rate of task. Using @property in case any future modifications
        decide to change fps on initialization
        '''
        return 1./self.fps        

    def print_to_terminal(self, *args):
        '''
        Print to the terminal rather than the websocket if the websocket is being used by the 'Notify' feature (see db.tasktrack)
        '''
        if len(args) == 1:
            print(args[0])
        else:
            print(args)        

    def init(self):
        '''Interface for child classes to run initialization code after object
        construction'''
        pass

    def run(self):
        '''
        Generic method to run the finite state machine of the task. Code that needs to execute 
        imediately before the task starts running in child classes should be of the form:

        def run(self):
            do stuff
            try:
                super(class_name, self).run()
            finally:
                clean up stuff

        The try block may or may not be necessary. For example, if you're opening a UDP port, you may want to always
        close the socket whether or not the main loop executes properly so that you don't loose the 
        reference to the socket. 
        '''

        ## Initialize the FSM before the loop
        self.set_state(self.state)
        
        while self.state is not None:
            if self.debug: 
                # allow ungraceful termination if in debugging mode so that pdb 
                # can catch the exception in the appropriate place
                self.fsm_tick()
            else:
                # in "production" mode (not debugging), try to capture & log errors gracefully
                try:
                    self.fsm_tick()
                except:
                    self.print_to_terminal("Error in FSM tick")
                    self.state = None
                    self.terminated_in_error = True

                    self.termination_err = io.StringIO()
                    traceback.print_exc(None, self.termination_err)
                    self.termination_err.seek(0)

                    self.print_to_terminal(self.termination_err.read())
                    self.termination_err.seek(0)
        if self.verbose: print("end of FSM.run, task state is", self.state)

    def run_sync(self):
        self.init()
        self.run()

    ###########################################################
    ##### Finite state machine (FSM) transition functions #####
    ###########################################################
    def fsm_tick(self):
        '''
        Execute the commands corresponding to a single tick of the event loop
        '''
    
        # Execute commands
        self.exec_state_specific_actions(self.state)

        current_state = self.state

        # iterate over the possible events which could move the task out of the current state
        for event in self.status[current_state]:
            if self.test_state_transition_event(event): # if the event has occurred
                # execute commands to end the current state
                self.end_state(current_state)

                # trigger the transition for the event
                self.trigger_event(event)

                # stop searching for transition events (transition events must be 
                # mutually exclusive for this FSM to function properly)
                break

        # Execute the commands which must run every loop, independent of the FSM state
        # (e.g., running the BMI decoder, updating the display)
        if self.state is not None:
            self._cycle()    


    def test_state_transition_event(self, event):
        event_test_fn_name = "_test_%s" % event
        if hasattr(self, event_test_fn_name):
            event_test_fn = getattr(self, event_test_fn_name)
            time_since_state_started = self.get_time() - self.start_time
            return event_test_fn(time_since_state_started)
        else:
            return False

    def end_state(self, state):
        end_state_fn_name = "_end_%s" % state
        if hasattr(self, end_state_fn_name):
            end_state_fn = getattr(self, end_state_fn_name)
            end_state_fn()

    def start_state(self, state):
        state_start_fn_name = "_start_%s" % state
        if hasattr(self, state_start_fn_name):
            state_start_fn = getattr(self, state_start_fn_name)
            state_start_fn()

    def exec_state_specific_actions(self, state):
        if hasattr(self, "_while_%s" % state):
            getattr(self, "_while_%s" % state)()

    def trigger_event(self, event):
        '''
        Transition the task state to a new state, where the next state depends on the current state as well as the trigger event

        Parameters
        ----------
        event: string
            Based on the current state, a particular event will trigger a particular state transition (Mealy machine)

        Returns
        -------
        None
        '''
        log = (self.state, event) not in self.log_exclude
        if log:  
            self.event_log.append((self.state, event, self.get_time()))

        fsm_edges = self.status[self.state]
        next_state = fsm_edges[event]            
        self.set_state(next_state, log=log)        
        
    def set_state(self, condition, log=True):
        '''
        Change the state of the task

        Parameters
        ----------
        condition: string
            Name of new state. The state name must be a key in the 'status' dictionary attribute of the task

        Returns
        -------
        None
        '''
        # Record the time at which the new state is entered. Used for timed states, e.g., the reward state
        self.start_time = self.get_time()

        if log:
            self.state_log.append((condition, self.start_time))        
        self.state = condition

        self.start_state(condition)

    def get_time(self):
        '''
        Abstraction to get the current time. By default, state transitions are based on wall clock time, not on iteration count.
        To get simulations to run faster than real time, this function must be overwritten.

        Returns
        -------
        float: The current time in seconds
        '''
        return time.time()

    def _cycle(self):
        '''
        Code that needs to run every task loop iteration goes here
        '''
        self.cycle_count += 1
        if self.fps > 0:
            self.clock.tick(self.fps)   

    def iter_time(self):
        '''
        Determine the time elapsed since the last time this function was called
        '''
        start_time = self.get_time()
        loop_time = start_time - self.last_time
        self.last_time = start_time
        return loop_time

    @classmethod
    def parse_fsm(cls):
        '''
        Print out the FSM of the task in a semi-readable form
        '''
        for state in cls.status:
            print('When in state "%s"' % state) 
            for trigger_event, next_state in list(cls.status[state].items()):
                print('\tevent "%s" moves the task to state "%s"' % (trigger_event, next_state))

    @classmethod
    def auto_gen_fsm_functions(cls):
        '''
        Parse the FSM to write all the _start, _end, _while, and _test functions
        '''
        events_to_test = []
        for state in cls.status:
            # make _start function 
            print('''def _start_%s(self): pass''' % state)

            # make _while function
            print('''def _while_%s(self): pass''' % state)
            # make _end function
            print('''def _end_%s(self): pass''' % state)
            for event, _ in cls.status.get_possible_state_transitions(state):
                events_to_test.append(event)

        print("################## State trnasition test functions ##################")

        for event in events_to_test:
            if event == 'stop': continue
            print('''def _test_%s(self, time_in_state): return False''' % event)

    def end_task(self):
        '''
        End the FSM gracefully on the next iteration by setting the task's "stop" flag.
        '''
        self.stop = True      

    def _test_stop(self, ts):
        ''' 
        FSM 'test' function. Returns the 'stop' attribute of the task
        '''
        return self.stop


class ThreadedFSM(FSM, threading.Thread):
    """ FSM + infrastructure to run FSM in its own thread """
    def __init__(self):
        FSM.__init__(self)
        threading.Thread.__init__(self)

    def start(self):
        '''
        From the python docs on threading.Thread:
            Once a thread object is created, its activity must be started by 
            calling the thread's start() method. This invokes the run() method in a 
            separate thread of control.

        Prior to the thread's start method being called, the secondary init function (self.init) is executed.
        After the threading.Thread.start is executed, the 'run' method is executed automatically in a separate thread.

        Returns
        -------
        None
        '''
        self.init()
        threading.Thread.start(self)

    def join(self):
        '''
        Code to run before re-joining the FSM thread 
        '''
        threading.Thread.join(self)
