import os
import time
import random
import threading

import traits.api as traits

import Pygame
import features

class Experiment(traits.HasTraits, threading.Thread):
    status = dict(
        wait = dict(start_trial="trial", premature="penalty", stop=None),
        trial = dict(correct="reward", incorrect="penalty", timeout="penalty"),
        reward = dict(restart="wait"),
        penalty = dict(restart="wait"),
    )
    state = "wait"
    stop = False

    def __init__(self, **kwargs):
        traits.HasTraits.__init__(self, **kwargs)
        threading.Thread.__init__(self)

    def trigger_event(self, event):
        self.set_state(self.status[self.state][event])
    
    def set_state(self, condition):
        print condition
        self.state = condition
        self.start_time = time.time()
        if hasattr(self, "_start_%s"%condition):
            getattr(self, "_start_%s"%condition)()

    def run(self):
        self.set_state(self.state)
        while self.state is not None:
            if hasattr(self, "_while_%s"%self.state):
                getattr(self, "_while_%s"%self.state)()
            
            for event, state in self.status[self.state].items():
                if hasattr(self, "_test_%s"%event):
                    if getattr(self, "_test_%s"%event)(time.time() - self.start_time):
                        self.trigger_event(event)
                        break;
                else:
                    print "can't find %s event"%event
    
    def _test_stop(self, ts):
        return self.stop
    
    def end_task(self):
        self.stop = True

class LogExperiment(Experiment):
    state_log = []
    event_log = []

    def trigger_event(self, event):
        self.event_log.append((self.state, event, time.time()))
        super(LogExperiment, self).trigger_event(event)

    def set_state(self, condition):
        self.state_log.append((condition, time.time()))
        super(LogExperiment, self).set_state(condition)

class TrialTypes(LogExperiment):
    trial_types = []
    status = dict(
        wait = dict(start_trial="picktrial", premature="penalty", stop=None),
        reward = dict(restart="wait"),
        penalty = dict(restart="wait"),
    )

    def __init__(self, **kwargs):
        super(TrialTypes, self).__init__(**kwargs)
        assert len(self.trial_types) > 0

        if self.trial_probs is None:
            self.trial_probs = [float(i+1) / len(self.trial_types) for i in range(len(self.trial_types))]
        elif any([i is None for i in self.trial_probs]):
            #Fix up the missing NONE entry
            assert sum([i is None for i in self.trial_probs]) == 1, "Too many None entries for probabilities, only one allowed!"
            prob = sum([i for i in self.trial_probs if i is not None])
            i = 0
            while self.trial_probs[i] is not None:
                i += 1
            self.trial_probs[i] = 1 - prob
        
        probs = self.trial_probs

        for ttype, (low, high) in zip(self.trial_types, probs):
            self.status[ttype] = {
                "%s_correct"%ttype:"reward", 
                "%s_incorrect"%ttype:"penalty", 
                "timeout":"penalty" }

            def func(self):
                return low <= self.trial_rand < high
            setattr(self, "_test_%s"%ttype, func)
    
    def _start_picktrial(self):
        self.trial_rand = random.random()

def make_experiment(exp_class, feats=()):
    allfeats = dict(
        button=features.Button,
        button_only=features.ButtonOnly,
        autostart=features.Autostart,
        ignore_correctness=features.IgnoreCorrectness
    )
    clslist = tuple(allfeats[f] for f in feats if f in allfeats)
    clslist = clslist + tuple(f for f in feats if f not in allfeats) + (exp_class,)
    return type(exp_class.__name__, clslist, dict())

def consolerun(exp_class, features=(), **kwargs):
    Class = make_experiment(exp_class, features)
    exp = Class(**kwargs)
    exp.start()
    raw_input("End? ")    
    exp.end_task()
    print "Waiting to end..."
    exp.join()
