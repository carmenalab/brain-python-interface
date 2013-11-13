'''
Experimental task base classes, contains mostly code to run the generic 
finite state machine representing different phases of the task
'''

import time
import random
import threading
import traceback
import collections

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

    def __init__(self, **kwargs):
        traits.HasTraits.__init__(self, **kwargs)
        threading.Thread.__init__(self)
        self.task_start_time = self.get_time()
        self.ntrials = 0
        self.nrewards = 0
        self.reward_len = 0
        self.reportstats = collections.OrderedDict()
        self.reportstats['State'] = None #State stat is automatically updated for all experiment classes
        self.reportstats['Runtime'] = '' #Runtime stat is automatically updated for all experiment classes
        self.reportstats['Trial #'] = 0 #Trial # stat must be updated by individual experiment classes
        self.reportstats['Reward #'] = 0 #Rewards stat is updated automatically for all experiment classes

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
                if hasattr(self, "_cycle"):
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

    
    def _test_stop(self, ts):
        return self.stop

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

    def cleanup(self, database, saveid, **kwargs):
        pass
    
    def end_task(self):
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
