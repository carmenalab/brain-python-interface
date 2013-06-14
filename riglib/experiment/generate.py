'''Needs docs'''

import random
import itertools

import numpy as np

from experiment import TrialTypes

def endless(exp, probs=None):
    if probs is None:
        while True:
            yield random.choice(exp.trial_types)
    else:
        assert len(probs) == len(exp.trial_types)
        probs = np.insert(np.cumsum(_fix_missing(probs)), 0, 0)
        assert probs[-1] == 1, "Probabilities do not add up to 1!"
        while True:
            rand = random.random()
            p = np.nonzero(rand < probs)[0].min()
            yield exp.trial_types[p-1]

def sequence(length, probs=2):
    '''Generates a sequence of numbers with the given probabilities.
    If probs is not a list, generate a uniformly distributed set of options.'''
    try:
        opts = len(probs)
        probs = _fix_missing(probs)
    except TypeError:
        opts = probs
        probs = [1 / float(opts)] * opts
    return np.random.permutation([i for i, p in enumerate(probs) for _ in xrange(int(length*p))])

def runseq(exp, seq=None, reps=1):
    if hasattr(exp, "trial_types"):
        assert max(seq)+1 == len(exp.trial_types)
        for _ in range(reps):
            for s in seq:
                yield exp.trial_types[s]
    else:
        for _ in range(reps):
            for s in seq:
                yield s

def _fix_missing(probs):
    '''Takes a probability list with possibly None entries, and fills it up'''
    total, n = map(sum, zip(*((i, 1) for i in probs if i is not None)))
    if n < len(probs):
        p = (1 - total) / (len(probs) - n)
        probs = [i or p for i in probs]
    return probs

class AdaptiveTrials(object):
    def __init__(self, exp, blocklen=8):
        assert issubclass(exp, TrialTypes)
        self.blocklen = blocklen
        self.trial_types = exp.trial_types
        self.new_block()

    def new_block(self):
        perblock = self.blocklen / len(self.trial_types)
        block = [[t]*perblock for t in self.trial_types]
        self.block = list(itertools.chain(*block))
        random.shuffle(self.block)

    def next(self):
        if len(self.block) < 1:
            self.new_block()
        return self.block[0]
    
    def correct(self):
        self.block.pop(0)
    
    def incorrect(self):
        ondeck = self.block.pop(0)
        self.block.append(ondeck)
        
