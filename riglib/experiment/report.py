import numpy as np
from experiment import LogExperiment, TrialTypes

def report(exp):
    assert isinstance(exp, LogExperiment), "Cannot report on non-logged experiments"
    ttypes = exp.trial_types if isinstance(exp, TrialTypes) else ["trial"]
    print ttypes
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
