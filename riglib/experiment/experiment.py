'''
Experimental task base classes, contains mostly code to run the generic
finite state machine representing different phases of the task
'''

import traceback
import collections
import re
import os
import tables
import traceback
import numpy as np
from collections import OrderedDict

from config.rig_defaults import rig_settings
from . import traits
from .. import fsm
from ..fsm import FSMTable, StateTransitions, ThreadedFSM

try:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
    pygame_avail = True
except ImportError:
    import warnings
    warnings.warn("experiment.py: Cannot import 'pygame'")
    pygame_avail = False


min_per_hour = 60.
sec_per_min = 60.

def time_to_string(sec):
    '''
    Convert a time in seconds to a string of format hh:mm:ss.
    '''
    nhours = int(sec/(min_per_hour * sec_per_min))
    nmins = int((sec-nhours*min_per_hour*sec_per_min)/sec_per_min)
    nsecs = int(sec - nhours*min_per_hour*sec_per_min - nmins*sec_per_min)
    return str(nhours).zfill(2) + ':' + str(nmins).zfill(2) + ':' + str(nsecs).zfill(2)

def _get_trait_default(trait):
    '''
    Function which tries to determine the default value for a trait in the class declaration
    '''
    _, default = trait.default_value()
    if isinstance(default, tuple) and len(default) > 0:
        try:
            func, args, _ = default
            default = func(*args)
        except:
            pass
    return default

class ExperimentMeta(type(traits.HasTraits)):
    '''
    Metaclass that merges traits and controls across class which
    inheritance from multiple parents
    '''
    def __new__(meta, name, bases, dct):
        '''
        Merge the parent trait lists into the class trait lists
        '''
        exclude_parent_traits = set()
        ordered_traits = set()
        hidden_traits = set()

        all_dct = [cls.__dict__ for cls in bases]
        all_dct.append(dct)
        for parent in all_dct:
            for key, value in parent.items():
                if key == 'hidden_traits':
                    hidden_traits.update(value)
                elif key == 'ordered_traits':
                    ordered_traits.update(value)
                elif key == 'exclude_parent_traits':
                    exclude_parent_traits.update(value)
        dct['exclude_parent_traits'] = exclude_parent_traits
        dct['ordered_traits'] = ordered_traits
        dct['hidden_traits'] = hidden_traits
        return super().__new__(meta, name, bases, dct)

    @property
    def controls(cls):
        '''
        Lookup the methods which are tagged with bmi3d_control.
        '''
        controls = set()
        for parent in cls.__mro__:
            for key, value in parent.__dict__.items():
                if hasattr(value, 'bmi3d_control') and value.bmi3d_control:
                    controls.add(value)
        return controls

def control_decorator(fn):
    fn.bmi3d_control = True
    return fn

class Experiment(ThreadedFSM, traits.HasTraits, metaclass=ExperimentMeta):
    '''
    Common ancestor of all task/experiment classes
    '''
    status = dict(
        wait = dict(start_trial="trial", premature="penalty", stop=None),
        trial = dict(correct="reward", incorrect="penalty", timeout="penalty"),
        reward = dict(post_reward="wait", end_state=True),
        penalty = dict(post_penalty="wait", end_state=True),
    )

    # Set the initial state to 'wait'. The 'wait' state has special behavior for the Sequence class (see below)
    state = "wait"

    # Flag to indicate that the task object has not been constructed or initialized
    _task_init_complete = False

    # Flag to set in order to stop the FSM gracefully
    stop = False

    # Rate at which FSM is called. Set to 60 Hz by default to match the typical monitor update rate
    fps = traits.Float(60, desc="Rate at which the FSM is called") # Hz

    cycle_count = 0

    # set this flag to true if certain things should only happen in debugging mode
    debug = False
    terminated_in_error = False

    ## GUI/database-related attributes
    # Flag to specify if you want to be able to create a BMI Decoder object from the web interface
    is_bmi_seed = False

    # Trait GUI manipulation
    exclude_parent_traits = [] # List of possible parent traits that you don't want to be set from the web interface
    ordered_traits = [] # Traits in this list appear in order at the top of the web interface parameters
    hidden_traits = []  # These traits are hidden on the web interface, and can be displayed by clicking the 'Show' radiobutton on the web interface

    # Runtime settable traits
    session_length = traits.Float(0, desc="Time until task automatically stops. Length of 0 means no auto stop.")

    # Initialization functions -----------------------------------------------
    @classmethod
    def pre_init(cls, **kwargs):
        '''
        Jobs to do before creating the task object go here (or this method should be overridden in child classes).
        Examples might include sending a trigger to start a recording device (e.g., neural system), since you might want
        recording to be guaranteed to start before any task event loop activity occurs.
        '''
        print('running experiment.Experiment.pre_init')
        pass

    def __init__(self, verbose=True, **kwargs):
        '''
        Constructor for Experiment. This is the standard python object constructor

        Parameters
        ----------
        kwargs: optional keyword-arguments
            Any user-specified parameters for experiment traits, to be passed to the traits.HasTraits parent.

        Returns
        -------
        Experiment instance
        '''
        traits.HasTraits.__init__(self, **kwargs)
        ThreadedFSM.__init__(self)
        self.verbose = verbose
        self.task_start_time = self.get_time()
        self.saveid = kwargs['saveid'] if 'saveid' in kwargs else None
        self.reportstats = collections.OrderedDict()
        self.reportstats['State'] = None # State stat is automatically updated for all experiment classes
        self.reportstats['Runtime'] = '' # Runtime stat is automatically updated for all experiment classes
        self.reportstats['Trial #'] = 0 # Trial # stat must be updated by individual experiment classes
        # If the FSM is set up in the old style (explicit dictionaries instead of wrapper data types), convert to the newer FSMTable
        if isinstance(self.status, dict):
            self.status = FSMTable.construct_from_dict(self.status)

        # Attribute for task entry dtype, used to create a numpy record array which is updated every iteration of the FSM
        # See http://docs.scipy.org/doc/numpy/user/basics.rec.html for details on how to create a record array dtype
        self.dtype = []

        self.cycle_count = 0
        if pygame_avail:
            self.clock = pygame.time.Clock()
        else:
            self.clock = fsm.Clock()

        self.pause = False


        ## Figure out which traits to not save to the HDF file
        ## Large/complex python objects cannot be saved as HDF file attributes
        ctraits = self.class_traits()
        self.object_trait_names = [ctr for ctr in list(ctraits.keys()) if ctraits[ctr].trait_type.__class__.__name__ in ['Instance', 'InstanceFromDB', 'DataFile']]

        if self.verbose: print("finished executing Experiment.__init__")

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

        # Create task_data record array
        # NOTE: all data variables MUST be declared prior to this point. So child classes overriding the 'init' method must
        # declare their variables using the 'add_dtype' function BEFORE calling the 'super' method.
        try:
            self.dtype = np.dtype(self.dtype)
            self.task_data = np.zeros((1,), dtype=self.dtype)
        except:
            print("Error creating the task_data record array")
            traceback.print_exc()
            print(self.dtype)
            self.task_data = None

        # Register the "task" source with the sinks
        if not hasattr(self, 'sinks'): # this attribute might be set in one of the other 'init' functions from other inherited classes
            from riglib import sink
            self.sinks = sink.SinkManager.get_instance()

        try:
            self.sinks.register("task", self.dtype, include_msgs=True)
        except:
            traceback.print_exc()
            raise Exception("Error registering task source")

        self._task_init_complete = True

    def add_dtype(self, name, dtype, shape):
        '''
        Add to the dtype of the task. The task's dtype attribute is used to determine
        which attributes to save to file.
        '''
        new_field = (name, dtype, shape)
        existing_field_names = [x[0] for x in self.dtype]
        if name in existing_field_names:
            raise Exception("Duplicate add_dtype functionc call for task data field: %s" % name)
        else:
            self.dtype.append(new_field)

    def screen_init(self):
        '''
        This method is implemented by the riglib.stereo_opengl.Window class, which is not used by all tasks. However,
        since Experiment is the ancestor of all tasks, a stub function is here so that any children
        using the window can safely use 'super'.
        '''
        pass

    def sync_event(self, event_name, event_data=None, immediate=False):
        '''
        Stub function for sending sync signals to various devices. Could be digital triggers to a recording system
        or markers on a screen measured by photodiode, for example. Implemented in features.sync_features
        '''
        pass 

        # TODO warning that you're not sending sync signals!

    # Trait functions --------------------------------------------------------
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
        # traits = super(Experiment, cls).class_editable_traits()
        from traits.trait_base import not_event, not_false
        traits = cls.class_trait_names(type=not_event, editable=not_false)
        editable_traits = [x for x in traits if x not in cls.exclude_parent_traits]
        return editable_traits

    @classmethod
    def get_trait_info(cls, trait_name, ctraits=None):
        """Get dictionary of information on a given trait"""
        if ctraits is None:
            ctraits = cls.class_traits()

        trait_params = dict()
        trait_params['type'] = ctraits[trait_name].trait_type.__class__.__name__
        trait_params['default'] = _get_trait_default(ctraits[trait_name])
        trait_params['desc'] = ctraits[trait_name].desc
        trait_params['hidden'] = 'hidden' if cls.is_hidden(trait_name) else 'visible'
        if hasattr(ctraits[trait_name], 'label'):
            trait_params['label'] = ctraits[trait_name].label
        else:
            trait_params['label'] = trait_name

        if trait_params['type'] == "InstanceFromDB":
            # a database instance. pass back the model and the query parameters and let the db
            # handle the rest
            trait_params['options'] = (ctraits[trait_name].bmi3d_db_model, ctraits[trait_name].bmi3d_query_kwargs)

        elif trait_params['type'] == 'Instance':
            raise ValueError("You should use the 'InstanceFromDB' trait instead of the 'Instance' trait!")

        elif trait_params['type'] == "Enum":
            raise ValueError("You should use the 'OptionsList' trait instead of the 'Enum' trait!")

        elif trait_params['type'] == "OptionsList":
            trait_params['options'] = ctraits[trait_name].bmi3d_input_options

        elif trait_params['type'] == "DataFile":
            trait_params['options'] = ("DataFile", ctraits[trait_name].bmi3d_query_kwargs)

        return trait_params

    @classmethod
    def get_params(cls):
        # Use an ordered dict so that params actually stay in the order they're added, instead of random (hash) order
        params = OrderedDict()

        ctraits = cls.class_traits()

        # add all the traits that are explicitly instructed to appear at the top of the menu
        ordered_traits = cls.ordered_traits
        for trait in ordered_traits:
            if trait in cls.class_editable_traits():
                params[trait] = cls.get_trait_info(trait, ctraits=ctraits)

        # add all the remaining non-hidden traits
        for trait in cls.class_editable_traits():
            if trait not in params and not cls.is_hidden(trait):
                params[trait] = cls.get_trait_info(trait, ctraits=ctraits)

        # add any hidden traits
        for trait in cls.class_editable_traits():
            if trait not in params:
                params[trait] = cls.get_trait_info(trait, ctraits=ctraits)
        return params

    def get_trait_values(self):
        '''
        Retrieve all the values of the 'trait' objects
        '''
        trait_values = dict()
        for trait in self.class_editable_traits():
            trait_values[trait] = getattr(self, trait)
        return trait_values

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

    @classmethod
    def get_desc(cls, params, report):
        return "An experiment!"

    # FSM functions ----------------------------------------------------------
    def run(self):
        '''
        Method to run the finite state machine of the task. Code that needs to execute
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
        self.reportstats['State'] = self.state
        super(Experiment, self).run()

    def _cycle(self):
        '''
        Code that needs to run every task loop iteration goes here
        '''
        super(Experiment, self)._cycle()

        # Send task data to any registered sinks
        if self.task_data is not None:
            self.sinks.send("task", self.task_data)

        # Update report stats periodically
        if self.cycle_count % self.fps == 0: 
            self.update_report_stats()

    def set_state(self, condition, *args, **kwargs):
        self.reportstats['State'] = condition or 'stopped'
        super().set_state(condition, *args, **kwargs)

    def _test_stop(self, ts):
        '''
        FSM 'test' function. Returns the 'stop' attribute of the task. Will only be
        called if the current state is 'stoppable', i.e. it has a 'stop' entry in its 
        state transition table.
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

    # UI interaction functions -----------------------------------------------
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
        values can be numbers or strings. Called on every state change.
        '''
        self.reportstats['Runtime'] = self._time_to_string(self.get_time() - self.task_start_time)

    def online_report(self):
        return self.reportstats

    def get_state(self):
        return self.state

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

    @staticmethod
    def log_summary(event_log):
        '''Return summary of trials in this experimental block
        exp_log: sequence of state transitions recorded by LogExperiment class
        '''
        report = dict()

        if len(event_log) == 0:
            report['runtime'] = 0
        else:
            report['runtime'] = event_log[-1][-1] - event_log[0][-1]

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

        report['n_trials'] = n_trials
        report['n_success_trials'] = n_success_trials
        return report

    @staticmethod
    def format_log_summary(report):
        '''Pretty-print the output of `log_summary`'''
        offline_report = collections.OrderedDict()
        offline_report['Runtime'] = time_to_string(report['runtime'])
        offline_report['Total trials'] = report['n_trials']
        offline_report['Total rewards'] = report['n_success_trials']

        # derived metrics
        n_success_trials = float(report['n_success_trials'])
        n_trials = float(report['n_trials'])
        if report['runtime'] > 0:
            rewards_per_min = np.round(n_success_trials/report['runtime'] * 60, decimals=2)
        else:
            rewards_per_min = 0

        if n_trials > 0:
            success_rate = np.round(n_success_trials/n_trials*100, decimals=2)
        else:
            success_rate = 0

        offline_report['Rewards/min'] = rewards_per_min
        offline_report['Success rate'] = '%g %%' % success_rate
        return offline_report

    # UI cleanup functions ---------------------------------------------------
    def cleanup(self, database, saveid, **kwargs):
        '''
        Commands to execute at the end of a task if it is to be saved to the database.

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
        success : bool
            False will trigger an error in the web GUI
        '''
        if self.verbose: print("experimient.Experiment.cleanup executing")
        return True

    def cleanup_hdf(self):
        '''
        Method for adding data to hdf file after hdf sink is closed by
        system at end of task. The HDF file is re-opened and any extra task
        data kept in RAM is written
        '''
        if hasattr(self, "h5file"):
            traits = self.class_editable_traits()
            h5file = tables.open_file(self.h5file.name, mode='a')
            
            # Create an empty task database if there isn't one already (may be empty if there was no task data)
            if not hasattr(h5file.root, "task"):
                h5file.create_table("/", "task", [('time', 'u8')])

            # Append to task data metadata
            for trait in traits:
                if (trait not in self.object_trait_names): # don't save traits which are complicated python objects to the HDF file
                    h5file.root.task.attrs[trait] = getattr(self, trait) 
            h5file.close()

    def terminate(self):
        '''
        Cleanup commands for all tasks regardless of whether they are saved or not
        '''
        pass

    # Web-facing controls --------------------------------------
    @control_decorator
    def play_pause(self):
        self.pause = not self.pause

        if 'pause' not in self.status.states:
            self.sync_event("PAUSE", immediate=True)

        if self.pause:
            print("Paused!")
        else:
            print("...resuming!")

        return self.pause


class LogExperiment(Experiment):
    '''
    Extension of the experiment class which logs state transitions
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reportstats['Success rate'] = "0 %"
        self.reportstats['Success rate (10 trials)'] = "0 %"

    def update_report_stats(self):
        super().update_report_stats()
        n_rewards = self.calc_state_occurrences('reward')
        n_trials = max(1, self.calc_trial_num())
        self.reportstats['Trial #'] = n_trials - 1
        self.reportstats['Success rate'] = "{} %".format(int(100*n_rewards/n_trials))
        self.reportstats['Success rate (10 trials)'] = "{} %".format(int(100*self.calc_events_per_trial("reward", 10)))

    def cleanup(self, database, saveid, **kwargs):
        '''
        Commands to execute at the end of a task.
        Save the task event log to the database

        see riglib.Experiment.cleanup for argument descriptions
        '''
        if self.verbose: print("experiment.LogExperiment.cleanup")
        super(LogExperiment, self).cleanup(database, saveid, **kwargs)
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'

        report_stats = self.log_summary(self.event_log)
        print("save_log", report_stats)

        if dbname == 'default':
            database.save_log(saveid, report_stats)
        else:
            database.save_log(saveid, report_stats, dbname=dbname)
        return True

    ##########################################################
    ##### Functions to calculate statistics from the log #####
    ##########################################################
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
        trialtimes = [state[1] for state in self.state_log if state[0] in self.status.trial_end_states]
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
        times = np.array([state[1] for state in self.state_log if state[0]==event_name])
        if (self.get_time() - self.task_start_time) < window:
            divideby = (self.get_time() - self.task_start_time)/sec_per_min
        else:
            divideby = window/sec_per_min
        return np.sum(times >= (self.get_time() - window))/divideby

    def calc_time_since_last_event(self, event_name):
        '''
        Calculates the time elapsed since the previous instance of event_name
        '''
        start_time = self.state_log[0][1]
        times = np.array([state[1] for state in self.state_log if state[0]==event_name])
        if len(times):
            return times[-1] - start_time
        else:
            return np.float64("0.0")

    def calc_events_per_trial(self, event_name, window):
        '''
        Calculates the rate of event_name, per trial

        Parameters
        ----------
        event_name: string
            Name of the state to calculate
        window: int
            Number of trials into the past to calculate the rate estimate
        
        Returns
        -------
        rate : float
            Rate of specified event, per trial
        '''
        trialtimes = [state[1] for state in self.state_log if state[0] in self.status.trial_end_states]
        if len(trialtimes) == 0:
            return 0
        elif len(trialtimes) < window:
            times = np.array([state[1] for state in self.state_log if state[0]==event_name and state[1] > trialtimes[0]])
            return len(times) / max(1, len(trialtimes) - 1)
        else:
            times = np.array([state[1] for state in self.state_log if state[0]==event_name and state[1] > trialtimes[-window]])
            return  len(times) / (window - 1)

class Sequence(LogExperiment):
    '''
    Task where the targets or other information relevant to the start of each trial
    are presented by a Python generator
    '''

    # List of staticmethods of the class which can be used to create a sequence of targets for each trial
    sequence_generators = []

    @classmethod
    def get_default_seq_generator(cls):
        '''
        Define a default sequence generator as the first one listed in the 'sequence_generators' attribute
        '''
        return getattr(cls, cls.sequence_generators[0])

    def __init__(self, gen=None, *args, **kwargs):
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
        if gen is None:
            raise ValueError("Experiment classes which inherit from Sequence must specify a target generator!")

        if hasattr(gen, '__next__'): # is iterable already
            self.gen = gen
        elif np.iterable(gen):
            from .generate import runseq
            self.gen = runseq(self, seq=gen)
        else:
            raise ValueError("Input argument to Sequence 'gen' must be of 'generator' type!")

        self.trial_dtype = np.dtype([('time', 'u8'), ('trial', 'u4')]) # to be overridden in init()

        super(Sequence, self).__init__(*args, **kwargs)

    def init(self, *args, **kwargs):

        # Create a record array for trials
        if not hasattr(self, 'sinks'): # this attribute might be set in one of the other 'init' functions from other inherited classes
            from riglib import sink
            self.sinks = sink.SinkManager.get_instance()
        dtype = self.trial_dtype # if you want to change this, do it in init() before calling super().init()
        self.trial_record = np.zeros((1,), dtype=dtype)
        self.sinks.register("trials", dtype)
        super().init(*args, **kwargs)

    def _start_wait(self):
        '''
        At the start of the wait state, the generator (self.gen) is querried for
        new information needed to start the trial. If the generator runs out, the task stops.
        '''
        if self.debug:
            print("_start_wait")

        try:
            self.next_trial = next(self.gen)
            # self._parse_next_trial() # KP changed 10/26/22
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

        self.trial_record['time'] = self.cycle_count
        self.trial_record['trial'] = self.calc_trial_num()
        self.sinks.send("trials", self.trial_record)


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
