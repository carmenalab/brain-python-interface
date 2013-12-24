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
import tables

import numpy as np
from . import traits

class Experiment(traits.HasTraits, threading.Thread):
    status = dict(
        wait = dict(start_trial="trial", premature="penalty", stop=None),
        trial = dict(correct="reward", incorrect="penalty", timeout="penalty"),
        reward = dict(post_reward="wait"),
        penalty = dict(post_penalty="wait"),
    )
    state = "wait"
    stop = False
    exclude_parent_traits = []
    ordered_traits = []

    def __init__(self, **kwargs):
        traits.HasTraits.__init__(self, **kwargs)
        threading.Thread.__init__(self)
        self.task_start_time = self.get_time()
        self.reportstats = collections.OrderedDict()
        self.reportstats['State'] = None #State stat is automatically updated for all experiment classes
        self.reportstats['Runtime'] = '' #Runtime stat is automatically updated for all experiment classes
        self.reportstats['Trial #'] = 0 #Trial # stat must be updated by individual experiment classes
        self.reportstats['Reward #'] = 0 #Rewards stat is updated automatically for all experiment classes

    @classmethod
    def class_editable_traits(cls):
        traits = super(Experiment, cls).class_editable_traits()
        editable_traits = filter(lambda x: x not in cls.exclude_parent_traits, traits)
        return editable_traits

    def init(self):
        '''
        Initialization method to run *after* object construction (see self.start)
        Over-ride in base class if there's anything to do just before the
        experiment starts running
        '''
        pass

    def screen_init(self):
        pass

    def trigger_event(self, event):
        '''
        Transition the task state, where the next state depends on the 
        trigger event
        '''
        self.set_state(self.status[self.state][event])

    def get_time(self):
        return time.time()

    def set_state(self, condition):
        self.state = condition
        self.start_time = self.get_time()
        self.update_report_stats()
        if hasattr(self, "_start_%s"%condition):
            getattr(self, "_start_%s"%condition)()

    def start(self):
        self.init()
        super(Experiment, self).start()

    def loop_step(self):
        '''
        Override this function to run some code every loop iteration of 
        the FSM
        '''
        pass

    def run(self):
        '''
        Generic method to run the finite state machine of the task
        '''
        self.screen_init()
        self.set_state(self.state)
        self.reportstats['State'] = self.state
        while self.state is not None:
            try:
                if hasattr(self, "_while_%s"%self.state):
                    getattr(self, "_while_%s"%self.state)()
                self._cycle()
                
                for event, state in self.status[self.state].items():
                    if hasattr(self, "_test_%s"%event):
                        if getattr(self, "_test_%s"%event)(self.get_time() - self.start_time):
                            if hasattr(self, "_end_%s"%self.state):
                                getattr(self, "_end_%s"%self.state)()
                            self.trigger_event(event)
                            break;
                self.loop_step()
            except:
                traceback.print_exc()
                self.state = None

    def _cycle(self):
        pass

    def _test_stop(self, ts):
        return self.stop

    def cleanup_hdf(self):
        ''' 
        Method for adding data to hdf file after hdf sink is closed by 
        system at end of task. The HDF file is re-opened and any extra task 
        data kept in RAM is written
        '''
        traits = self.class_editable_traits()
        h5file = tables.openFile(self.h5file.name, mode='a')
        for trait in traits:
            if trait not in ['bmi', 'arm_class', 'arm_visible']:
                #self.hdf.sendAttr("task", trait, )
                print trait
                h5file.root.task.attrs[trait] = getattr(self, trait)
        h5file.close()

    @classmethod
    def _time_to_string(self, sec):
        '''
        Convert a time in seconds to a string of format hh:mm:ss.
        '''
        nhours = int(sec/3600)
        nmins = int((sec-nhours*3600)/60)
        nsecs = int(sec - nhours*3600 - nmins*60)
        return str(nhours).zfill(2) + ':' + str(nmins).zfill(2) + ':' + str(nsecs).zfill(2)

    def update_report_stats(self):
        '''Function to update any relevant report stats for the task. Values are saved in self.reportstats,
        an ordered dictionary. Keys are strings that will be displayed as the label for the stat in the web interface,
        values can be numbers or strings. Called every time task state changes.'''
        self.reportstats['Runtime'] = self._time_to_string(self.get_time() - self.task_start_time)

    @classmethod
    def offline_report(self, event_log):
        '''Returns an ordered dict with report stats to be displayed when past session of this task is selected
        in the web interface. Not called while task is running, only offline, so stats must come from information
        available in a sessions event log. Inputs are task object and event_log.'''
        offline_report = collections.OrderedDict()  
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
        if n_trials == 0:
            offline_report['Success rate'] = None
        else:
            offline_report['Success rate'] = str(np.round(float(n_success_trials)/n_trials*100,decimals=2)) + '%'
        return offline_report

    def cleanup(self, database, saveid, **kwargs):
        pass
    
    def end_task(self):
        import logging
        logging.debug("Calling Experiment.end_task")
        self.stop = True

class LogExperiment(Experiment):
    log_exclude = set()
    def __init__(self, **kwargs):
        self.state_log = []
        self.event_log = []
        super(LogExperiment, self).__init__(**kwargs)
    
    def trigger_event(self, event):
        log = (self.state, event) not in self.log_exclude
        if log:  
            self.event_log.append((self.state, event, self.get_time()))
        self.set_state(self.status[self.state][event], log=log)

    def set_state(self, condition, log=True):
        if log:
            self.state_log.append((condition, self.get_time()))
        super(LogExperiment, self).set_state(condition)

    def cleanup(self, database, saveid, **kwargs):
        super(LogExperiment, self).cleanup(database, saveid, **kwargs)
        database.save_log(saveid, self.event_log)

class Sequence(LogExperiment):
    def __init__(self, gen, **kwargs):
        self.gen = gen
        assert hasattr(gen, "next"), "gen must be a generator"
        super(Sequence, self).__init__(**kwargs)
        #self.next_trial = self.gen.next()
    
    def _start_wait(self):
        try:
            self.next_trial = self.gen.next()
        except StopIteration:
            self.end_task()

class TrialTypes(Sequence):
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
