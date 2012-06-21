import time
import random
import threading

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

    def trigger_event(self, event):
        self.set_state(self.status[self.state][event])
    
    def set_state(self, condition):
        self.state = condition
        self.start_time = time.time()
        if hasattr(self, "_start_%s"%condition):
            getattr(self, "_start_%s"%condition)()

    def run(self):
        self.set_state(self.state)
        while self.state is not None:
            if hasattr(self, "_while_%s"%self.state):
                getattr(self, "_while_%s"%self.state)()
            if hasattr(self, "_cycle"):
                self._cycle()
            
            for event, state in self.status[self.state].items():
                if hasattr(self, "_test_%s"%event):
                    if getattr(self, "_test_%s"%event)(time.time() - self.start_time):
                        if hasattr(self, "_end_%s"%self.state):
                            getattr(self, "_end_%s"%self.state)()
                        self.trigger_event(event)
                        break;
    
    def _test_stop(self, ts):
        return self.stop
    
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
            self.event_log.append((self.state, event, time.time()))
        self.set_state(self.status[self.state][event], log=log)

    def set_state(self, condition, log=True):
        if log:
            self.state_log.append((condition, time.time()))
        super(LogExperiment, self).set_state(condition)

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
