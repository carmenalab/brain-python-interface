'''
Experimental task base classes, contains mostly code to run the generic 
finite state machine representing different phases of the task
'''

import time
import random
import threading
import traceback
import collections
import re
import os
import tables
import traceback

import numpy as np
from . import traits

try:
    import pygame
except ImportError:
    import warnings
    warnings.warn("experiment.py: Cannot import 'pygame'")

from collections import OrderedDict

min_per_hour = 60.
sec_per_min = 60.

class FSMTable(object):
    def __init__(self, **kwargs):
        self.states = OrderedDict()
        for state_name, transitions in kwargs.items():
            self.states[state_name] = transitions

    def __getitem__(self, key):
        return self.states[key]

    def get_possible_state_transitions(self, current_state):
        return self.states[current_state].items()

    def _lookup_next_state(self, current_state, transition_event):
        return self.states[current_state][transition_event]

    def __iter__(self):
        return self.states.keys().__iter__()

    @staticmethod
    def construct_from_dict(status):
        outward_transitions = OrderedDict()
        for state in status:
            outward_transitions[state] = StateTransitions(stoppable=False, **status[state])
        return FSMTable(**outward_transitions)


class StateTransitions(object):
    def __init__(self, stoppable=True, **kwargs):
        self.state_transitions = OrderedDict()
        for event, next_state in kwargs.items():
            self.state_transitions[event] = next_state

        if stoppable and not ('stop' in self.state_transitions):
            self.state_transitions['stop'] = None

    def __getitem__(self, key):
        return self.state_transitions[key]

    def __iter__(self):
        transition_events = self.state_transitions.keys()
        return transition_events.__iter__()

    def items(self):
        return self.state_transitions.items()


class Experiment(traits.HasTraits, threading.Thread):
    '''
    Common ancestor of all task/experiment classes
    '''
    status = dict(
        wait = dict(start_trial="trial", premature="penalty", stop=None),
        trial = dict(correct="reward", incorrect="penalty", timeout="penalty"),
        reward = dict(post_reward="wait"),
        penalty = dict(post_penalty="wait"),
    )

    # For analysis purposes, it's useful to declare which task states are "terminal" states and signify the end of a trial
    trial_end_states = []

    # Set the initial state to 'wait'. The 'wait' state has special behavior for the Sequence class (see below)
    state = "wait"

    # Flag to set in order to stop the FSM gracefully
    stop = False

    # Rate at which FSM is called. Set to 60 Hz by default to match the typical monitor update rate
    fps = 60 # Hz

    # set this flag to true if certain things should only happen in debugging mode
    debug = False

    ## GUI/database-related attributes
    # Flag to specify if you want to be able to create a BMI Decoder object from the web interface
    is_bmi_seed = False

    # Trait GUI manipulation
    exclude_parent_traits = [] # List of possible parent traits that you don't want to be set from the web interface
    ordered_traits = [] # Traits in this list appear in order at the top of the web interface parameters
    hidden_traits = []  # These traits are hidden on the web interface, and can be displayed by clicking the 'Show' radiobutton on the web interface

    # Runtime settable traits
    session_length = traits.Float(0, desc="Time until task automatically stops. Length of 0 means no auto stop.")

    @property 
    def update_rate(self):
        '''
        Attribute for update rate of task. Using @property in case any future modifications
        decide to change fps on initialization
        '''
        return 1./self.fps        

    @classmethod
    def class_editable_traits(cls):
        '''
        Class method to retrieve the list of editable traits for the given experiment. 
        The default behavior for an experiment class is to make all traits editable except for those
        listed in the attribute 'exclude_parent_traits'. 

        Parameters
        ----------
        None

        Returns
        -------
        editable_traits: list of strings
            Names of traits which are designated to be runtime-editable
        '''
        traits = super(Experiment, cls).class_editable_traits()
        editable_traits = filter(lambda x: x not in cls.exclude_parent_traits, traits)
        return editable_traits

    @classmethod
    def parse_fsm(cls):
        '''
        Print out the FSM of the task in a semi-readable form
        '''
        for state in cls.status:
            print 'When in state "%s"' % state 
            for trigger_event, next_state in cls.status[state].items():
                print '\tevent "%s" moves the task to state "%s"' % (trigger_event, next_state)

    @classmethod
    def auto_gen_fsm_functions(cls):
        '''
        Parse the FSM to write all the _start, _end, _while, and _test functions
        '''
        events_to_test = []
        for state in cls.status:
            # make _start function 
            print '''def _start_%s(self): pass''' % state

            # make _while function
            print '''def _while_%s(self): pass''' % state
            # make _end function
            print '''def _end_%s(self): pass''' % state
            for event, _ in cls.status.get_possible_state_transitions(state):
                events_to_test.append(event)

        print "################## State trnasition test functions ##################"

        for event in events_to_test:
            if event == 'stop': continue
            print '''def _test_%s(self, time_in_state): return False''' % event

    @classmethod
    def is_hidden(cls, trait):
        '''
        Return true if the given trait is not meant to be shown on the GUI by default, i.e. hidden 

        Parameters
        ----------
        trait: string
            Name of trait to check

        Returns
        -------
        bool
        '''
        return trait in cls.hidden_traits

    ####################################
    ##### Initialization functions #####
    ####################################
    @classmethod 
    def pre_init(cls, **kwargs):
        '''
        Jobs to do before creating the task object go here (or this method should be overridden in child classes)
        '''
        print 'running experiment.Experiment.pre_init'
        pass

    def __init__(self, **kwargs):
        '''
        Constructor for Experiment

        Parameters
        ----------
        kwargs: dictionary
            Keyword arguments to be passed to the traits.HasTraits parent.

        Returns
        -------
        Experiment instance
        '''
        traits.HasTraits.__init__(self, **kwargs)
        threading.Thread.__init__(self)
        self.task_start_time = self.get_time()
        self.reportstats = collections.OrderedDict()
        self.reportstats['State'] = None #State stat is automatically updated for all experiment classes
        self.reportstats['Runtime'] = '' #Runtime stat is automatically updated for all experiment classes
        self.reportstats['Trial #'] = 0 #Trial # stat must be updated by individual experiment classes
        self.reportstats['Reward #'] = 0 #Rewards stat is updated automatically for all experiment classes

        # If the FSM is set up in the old style (explicit dictionaries instead of wrapper data types), convert to the newer FSMTable
        if isinstance(self.status, dict):
            self.status = FSMTable.construct_from_dict(self.status)

        # Attribute for task entry dtype, used to create a numpy record array which is updated every iteration of the FSM
        # See http://docs.scipy.org/doc/numpy/user/basics.rec.html for details on how to create a record array dtype
        self.dtype = []

        self.cycle_count = 0
        self.clock = pygame.time.Clock()

        self.pause = False

        print "finished executing Experiment.__init__"

    def init(self):
        '''
        Initialization method to run *after* object construction (see self.start)
        Over-ride in base class if there's anything to do just before the
        experiment starts running
        '''
        # Timestamp for rough loop timing
        self.last_time = self.get_time()
        self.cycle_count = 0

        # Create task_data record array
        # NOTE: all data variables MUST be declared prior to this point. So child classes overriding the 'init' method must
        # declare their variables using the 'add_dtype' function BEFORE calling the 'super' method.
        try:
            if len(self.dtype) > 0:
                self.dtype = np.dtype(self.dtype)
                self.task_data = np.zeros((1,), dtype=self.dtype)
            else:
                self.task_data = None
        except:
            print "Error creating the task_data record array"
            traceback.print_exc()
            print self.dtype
            self.task_data = None

        # Register the "task" source with the sinks
        if not hasattr(self, 'sinks'): # this attribute might be set in one of the other 'init' functions from other inherited classes
            from riglib import sink
            self.sinks = sink.sinks

        try:
            self.sinks.register("task", self.dtype)
        except:
            traceback.print_exc()            
            raise Exception("Error registering task source")

    def add_dtype(self, name, dtype, shape):
        '''
        Add to the dtype of the task. The task's dtype attribute is used to determine 
        which attributes to save to file. 
        '''
        self.dtype.append((name, dtype, shape))

    def screen_init(self):
        '''
        This method is implemented by the riglib.stereo_opengl.Window class, which is not used by all tasks. However, 
        since Experiment is the ancestor of all tasks, a stub function is here so that any children
        using the window can safely use 'super'. 
        '''
        pass

    ###############################
    ##### Threading functions #####
    ###############################
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
        super(Experiment, self).start()

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
        self.reportstats['State'] = self.state
        
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
                    traceback.print_exc(open(os.path.expandvars('$BMI3D/log/exp_run_log'), 'w'))
                    self.state = None

    def run_sync(self):
        self.init()
        self.run()

    ###########################################################
    ##### Finite state machine (FSM) transition functions #####
    ###########################################################
    def fsm_tick(self):
        # Execute commands
        self.exec_state_specific_actions(self.state)

        # Execute the commands which must run every loop, independent of the FSM state
        # (e.g., running the BMI)
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

        # Update the report for the GUI
        self.update_report_stats()

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
        # print "Experiment._cycle"
        self.cycle_count += 1
        if self.fps > 0:
            self.clock.tick(self.fps)

        # Send task data to any registered sinks
        if self.task_data is not None:
            self.sinks.send("task", self.task_data)        

    def iter_time(self):
        '''
        Determine the time elapsed since the last time this function was called
        '''
        start_time = self.get_time()
        loop_time = start_time - self.last_time
        self.last_time = start_time
        return loop_time

    ##############################
    ##### FSM test functions #####
    ##############################
    def _test_stop(self, ts):
        ''' 
        FSM 'test' function. Returns the 'stop' attribute of the task
        '''
        if self.session_length > 0 and (self.get_time() - self.task_start_time) > self.session_length:
            self.end_task()
        return self.stop

    def _test_time_expired(self, ts):
        '''
        Generic function to test if time has expired. For a state 'wait', the function looks up the 
        variable 'wait_time' and uses that as a time.
        '''
        state_time_var_name = self.state + '_time'
        try:
            state_time = getattr(self, state_time_var_name)
        except AttributeError:
            raise AttributeError("Cannot find attribute %s; may not be able to use generic time_expired event for state %s"  % (state_time_var_name, self.state))

        assert isinstance(state_time, (float, int))
        return ts > state_time

    ############################
    ##### Report functions #####
    ############################
    @classmethod
    def _time_to_string(cls, sec):
        '''
        Convert a time in seconds to a string of format hh:mm:ss.
        '''
        nhours = int(sec/(min_per_hour * sec_per_min))
        nmins = int((sec-nhours*min_per_hour*sec_per_min)/sec_per_min)
        nsecs = int(sec - nhours*min_per_hour*sec_per_min - nmins*sec_per_min)
        return str(nhours).zfill(2) + ':' + str(nmins).zfill(2) + ':' + str(nsecs).zfill(2)

    def update_report_stats(self):
        '''
        Function to update any relevant report stats for the task. Values are saved in self.reportstats,
        an ordered dictionary. Keys are strings that will be displayed as the label for the stat in the web interface,
        values can be numbers or strings. Called every time task state changes.
        '''
        self.reportstats['Runtime'] = self._time_to_string(self.get_time() - self.task_start_time)

    @classmethod
    def offline_report(self, event_log):
        '''Returns an ordered dict with report stats to be displayed when past session of this task is selected
        in the web interface. Not called while task is running, only offline, so stats must come from information
        available in a sessions event log. Inputs are task object and event_log.'''
        offline_report = collections.OrderedDict()  
        if len(event_log) == 0:
            explength = 0
        else:
            explength = event_log[-1][-1] - event_log[0][-1]
        offline_report['Runtime'] = self._time_to_string(explength)
        n_trials = 0
        n_success_trials = 0
        n_error_trials = 0
        for k, (state, event, t) in enumerate(event_log):
            if state == "reward":
                n_trials += 1
                n_success_trials += 1
            elif re.match('.*?_penalty', state):
                n_trials += 1
                n_error_trials += 1
        offline_report['Total trials'] = n_trials
        offline_report['Total rewards'] = n_success_trials
        try:
            offline_report['Rewards/min'] = np.round((n_success_trials/explength) * 60, decimals=2)
        except:
            offline_report['Rewards/min'] = 0
        if n_trials == 0:
            offline_report['Success rate'] = None
        else:
            offline_report['Success rate'] = str(np.round(float(n_success_trials)/n_trials*100,decimals=2)) + '%'
        return offline_report

    def print_to_terminal(self, *args):
        '''
        Print to the terminal rather than the websocket if the websocket is being used by the 'Notify' feature (see db.tasktrack)
        '''
        print args

    ################################
    ## Cleanup/termination functions
    ################################
    def get_trait_values(self):
        '''
        Retrieve all the values of the 'trait' objects
        '''
        trait_values = dict()
        for trait in self.class_editable_traits():
            trait_values[trait] = getattr(self, trait)
        return trait_values

    def cleanup(self, database, saveid, **kwargs):
        '''
        Commands to execute at the end of a task.

        Parameters
        ----------
        database : object
            Needs to have the methods save_bmi, save_data, etc. For instance, the db.tracker.dbq module or an RPC representation of the database
        saveid : int
            TaskEntry database record id to link files/data to
        kwargs : optional dict arguments
            Optional arguments to dbq methods. NOTE: kwargs cannot be used when 'database' is an RPC object.

        Returns
        -------
        None
        '''
        print "experimient.Experiment.cleanup executing"
    
    def cleanup_hdf(self):
        ''' 
        Method for adding data to hdf file after hdf sink is closed by 
        system at end of task. The HDF file is re-opened and any extra task 
        data kept in RAM is written
        '''
        traits = self.class_editable_traits()

        if hasattr(tables, 'open_file'): # function name depends on version
            h5file = tables.open_file(self.h5file.name, mode='a')        
        else:
            h5file = tables.openFile(self.h5file.name, mode='a')
        for trait in traits:
            if trait not in ['bmi', 'decoder', 'ref_trajectories']:
                h5file.root.task.attrs[trait] = getattr(self, trait)
        h5file.close()

    def end_task(self):
        '''
        End the FSM gracefully on the next iteration by setting the task's "stop" flag.
        '''
        self.stop = True

    def terminate(self):
        '''
        Cleanup commands for tasks executed using the "test" button
        '''
        pass


class LogExperiment(Experiment):
    '''
    Extension of the experiment class which logs state transitions
    '''
    # List out state/trigger pairs to exclude from logging
    log_exclude = set()
    trial_end_states = []
    def __init__(self, **kwargs):
        '''
        Constructor for LogExperiment

        Parameters
        ----------
        kwargs: dict
            These are all propagated to the parent (none used for this constructor)

        Returns
        -------
        LogExperiment instance
        '''
        self.state_log = []
        self.event_log = []
        super(LogExperiment, self).__init__(**kwargs)
    
    def trigger_event(self, event):
        '''
        see riglib.Experiment.trigger_event for description.
        Saves a history of state transitions before executing the event
        '''
        log = (self.state, event) not in self.log_exclude
        if log:  
            self.event_log.append((self.state, event, self.get_time()))
        self.set_state(self.status[self.state][event], log=log)

    def set_state(self, condition, log=True):
        '''
        see riglib.Experiment.set_state for description.
        Saves the sequence of entered states before executing the parent's 'set_state'
        '''
        if log:
            self.state_log.append((condition, self.get_time()))
        super(LogExperiment, self).set_state(condition)

    def cleanup(self, database, saveid, **kwargs):
        '''
        Commands to execute at the end of a task. 
        Save the task event log to the database

        see riglib.Experiment.cleanup for argument descriptions
        '''
        print "experiment.LogExperiment.cleanup"
        super(LogExperiment, self).cleanup(database, saveid, **kwargs)
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if dbname == 'default':
            database.save_log(saveid, self.event_log)
        else:
            database.save_log(saveid, self.event_log, dbname=dbname)

    def calc_state_occurrences(self, state_name):
        '''
        Calculate the number of times the task enters a particular state

        Parameters
        ----------
        state_name: string
            Name of state to track

        Returns
        -------
        Counts of state occurrences 
        '''
        times = np.array([state[1] for state in self.state_log if state[0] == state_name])
        return len(times)

    def calc_trial_num(self):
        '''
        Counts the number of trials which have finished.
        '''
        trialtimes = [state[1] for state in self.state_log if state[0] in self.trial_end_states]
        return len(trialtimes)

    def calc_events_per_min(self, event_name, window):
        '''
        Calculates the rate of event_name, per minute

        Parameters
        ----------
        event_name: string
            Name of state representing "event"
        window: float
            Number of seconds into the past to look to calculate the current event rate estimate.

        Returns
        -------
        rate : float
            Rate of specified event, per minute
        '''
        rewardtimes = np.array([state[1] for state in self.state_log if state[0]==event_name])
        if (self.get_time() - self.task_start_time) < window:
            divideby = (self.get_time() - self.task_start_time)/sec_per_min
        else:
            divideby = window/sec_per_min
        return np.sum(rewardtimes >= (self.get_time() - window))/divideby

class Sequence(LogExperiment):
    '''
    Task where the targets or other information relevant to the start of each trial
    are presented by a Python generator
    '''

    # List of staticmethods of the class which can be used to create a sequence of targets for each trial
    sequence_generators = []

    @classmethod 
    def get_default_seq_generator(cls):
        return getattr(cls, cls.sequence_generators[0])

    def __init__(self, gen, *args, **kwargs):
        '''
        Constructor for Sequence

        Parameters
        ----------
        gen : Python generator
            Object with a 'next' attribute used in the special "wait" state to get the target sequence for the next trial.
        kwargs: optonal keyword-arguments
            Passed to the super constructor

        Returns
        -------
        Sequence instance
        '''
        if np.iterable(gen):
            from generate import runseq
            gen = runseq(self, seq=gen)

        self.gen = gen
        if not hasattr(gen, "next"):
            raise ValueError("Input argument to Sequence 'gen' must be of 'generator' type!")

        super(Sequence, self).__init__(*args, **kwargs)
    
    def _start_wait(self):
        '''
        At the start of the wait state, the generator (self.gen) is querried for 
        new information needed to start the trial. If the generator runs out, the task stops. 
        '''
        try:
            self.next_trial = self.gen.next()
        except StopIteration:
            self.end_task()

        self._parse_next_trial()
        
    def _parse_next_trial(self):
        '''
        Interpret the data coming from the generator. If the generator yields a dictionary, 
        then the keys of the dictionary automatically get set as attributes.

        Over-ride or add additional code in child classes if different behavior is desired.
        '''
        if isinstance(self.next_trial, dict):
            for key in self.next_trial:
                setattr(self, '_gen_%s' % key, self.next_trial[key])


class TrialTypes(Sequence):
    '''
    This module is deprecated, used by some older tasks (dots, rds)
    '''
    trial_types = []
        
    status = dict(
        wait = dict(start_trial="picktrial", premature="penalty", stop=None),
        reward = dict(post_reward="wait"),
        penalty = dict(post_penalty="wait"),
    )

    def __init__(self, gen, **kwargs):
        super(TrialTypes, self).__init__(gen, **kwargs)
        assert len(self.trial_types) > 0

        for ttype in self.trial_types:
            self.status[ttype] = {
                "%s_correct"%ttype :"reward", 
                "%s_incorrect"%ttype :"incorrect", 
                "timeout":"incorrect" }
            #Associate all trial type endings to the end_trial function defined by Sequence
            #setattr(self, "_end_%s"%ttype, self._end_trial)
    
    def _start_picktrial(self):
        self.set_state(self.next_trial)
    
    def _start_incorrect(self):
        self.set_state("penalty")
