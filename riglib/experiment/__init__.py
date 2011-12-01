import os
import time
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
    pass

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
