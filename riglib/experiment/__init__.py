import os
import time
import random
import threading
import numpy as np

try:
    import traits.api as traits
except ImportError:
    import enthought.traits.api as traits

import Pygame
import features

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
    trial_probs = None
    status = dict(
        wait = dict(start_trial="picktrial", premature="penalty", stop=None),
        reward = dict(post_reward="wait"),
        penalty = dict(post_penalty="wait"),
    )

    def __init__(self, **kwargs):
        super(TrialTypes, self).__init__(**kwargs)
        assert len(self.trial_types) > 0

        if self.trial_probs is None:
            self.trial_probs = [1./len(self.trial_types)] * len(self.trial_types)
        elif any([i is None for i in self.trial_probs]):
            #Fix up the missing NONE entry
            assert sum([i is None for i in self.trial_probs]) == 1, "Too many None entries for probabilities, only one allowed!"
            prob = sum([i for i in self.trial_probs if i is not None])
            i = 0
            while self.trial_probs[i] is not None:
                i += 1
            self.trial_probs[i] = 1 - prob
        probs = np.insert(np.cumsum(self.trial_probs), 0, 0)
        assert probs[-1] == 1
        self.probs = np.array([probs[:-1], probs[1:]]).T

        for ttype in self.trial_types:
            self.status[ttype] = {
                "%s_correct"%ttype :"reward", 
                "%s_incorrect"%ttype :"penalty", 
                "timeout":"penalty" }
    
    def _start_picktrial(self):
        rand = random.random()
        for i, (low, high) in enumerate(self.probs):
            if low <= rand < high:
                self.set_state(self.trial_types[i])
                break;

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
    while raw_input().strip() != "q":
        print_report(report(exp))
    exp.end_task()
    print "Waiting to end..."
    exp.join()
    return exp

def report(exp):
    assert isinstance(exp, LogExperiment), "Cannot report on non-logged experiments"
    ttypes = exp.trial_types if isinstance(exp, TrialTypes) else ["trial"]
    trials = dict([(t, dict(correct=[], incorrect=[], timeout=[])) for t in ttypes])
    report = dict(rewards=0, prematures=[], trials=trials)
    trial = None

    for state, event, t in exp.event_log:
        if trial is not None:
            if "incorrect" in event:
                report['trials'][state]["incorrect"].append(t - trial)
            elif "correct" in event:
                report['trials'][state]["correct"].append(t - trial)
            elif "timeout" in event:
                report['trials'][state]["timeout"].append(t - trial)
            trial = None
        
        if event == "start_trial":
            trial = t
        elif state == "reward":
            report['rewards'] += 1
    
    return report

def print_report(report):
    '''Prints a report generated by report(exp)'''
    repstr = ["%8s: %d"%("rewards", report['rewards'])]
    ttrial = 0
    for tname, tdict in report['trials'].items():
        total = len(tdict['correct']) + len(tdict['incorrect']) + len(tdict['timeout'])
        ttrial += total
        if total == 0:
            cper = 0
        else:
            cper = float(len(tdict['correct'])) / total * 100
        cRT = np.mean(tdict['correct'])
        repstr.append("%8s: %g%%, RT=%g"%(tname, cper, cRT))
    repstr.insert(0, "%8s: %d"%("total", ttrial))
    print "\n".join(repstr)
