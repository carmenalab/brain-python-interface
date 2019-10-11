"""Finite state machine implementation """
import time
import random
import threading
import traceback
import collections
import re
import os
import tables
import traceback
import io
import numpy as np

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

    @staticmethod
    def construct_from_dict(status):
        outward_transitions = OrderedDict()
        for state in status:
            outward_transitions[state] = StateTransitions(stoppable=False, **status[state])
        return FSMTable(**outward_transitions)


class StateTransitions(object):
    def __init__(self, stoppable=True, **kwargs):
        self.state_transitions = OrderedDict()
        for event, next_state in list(kwargs.items()):
            self.state_transitions[event] = next_state

        if stoppable and not ('stop' in self.state_transitions):
            self.state_transitions['stop'] = None

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

    def __init__(self, *args, **kwargs):
        self.clock = Clock()

    def screen_init(self):
        '''
        This method is implemented by the riglib.stereo_opengl.Window class, which is not used by all tasks. However, 
        since Experiment is the ancestor of all tasks, a stub function is here so that any children
        using the window can safely use 'super'. 
        '''
        pass

    def print_to_terminal(self, *args):
        '''
        Print to the terminal rather than the websocket if the websocket is being used by the 'Notify' feature (see db.tasktrack)
        '''
        if len(args) == 1:
            print(args[0])
        else:
            print(args)        

    def init(self):
        '''
        Initialization method to run *after* object construction (see self.start). 
        This may be necessary in some cases where features are used with multiple inheritance to extend tasks 
        (this is the standard way of creating custom base experiment + features classes through the browser interface). 
        With multiple inheritance, it's difficult/annoying to make guarantees about the order of operations for 
        each of the individual __init__ functions from each of the parents. Instead, this function runs after all the 
        __init__ functions have finished running if any subsequent initialization is necessary before the main event loop 
        can execute properly. Examples include initialization of the decoder state/parameters. 
        '''
        # Timestamp for rough loop timing
        self.last_time = self.get_time()
        self.cycle_count = 0        

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
        self.screen_init()
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

                    traceback.print_exc()
        print("end of experiment.run, task state is", self.state)

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

        # Execute the commands which must run every loop, independent of the FSM state
        # (e.g., running the BMI decoder)
        self._cycle()

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
        fsm_edges = self.status[self.state]
        next_state = fsm_edges[event]
        self.set_state(next_state)

    def set_state(self, condition):
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
        self.state = condition

        # Record the time at which the new state is entered. Used for timed states, e.g., the reward state
        self.start_time = self.get_time()

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