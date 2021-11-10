'''
Various generic generators to combine with tasks. These appear to be mostly deprecated at this point
'''

import random
import itertools

import numpy as np

from .experiment import TrialTypes, Sequence


def block_random(*args, **kwargs):
    '''
    A generic block randomizer. 

    Parameters
    ----------

    Returns
    -------
    seq: list
        Block-random sequence of items where the length of each block is the product of the length of each the parameters being varied
    '''
    n_blocks = kwargs.pop('nblocks')
    inds = [np.arange(len(arg)) for arg in args]
    from itertools import product
    items = []
    for x in product(*inds):
        item = [arg[i] for arg,i in zip(args, x)]
        items.append(item)

    n_items = len(items)
    seq = []
    for k in range(n_blocks):
        inds = np.arange(n_items)
        np.random.shuffle(inds)
        for i in inds:
            seq.append(items[i])

    return seq

def runseq(exp, seq=None, reps=1):
    '''
    Turns a sequence into a Python generator by iterating through the sequence and yielding each sequence element

    Parameters
    ----------
    exp: Class object for task
        Used only if the experiment has 'trial_types' instead of a target sequence list
    seq: iterable
        Target sequence yielded to the task during the 'wait' state
    reps: int, optional, default=1
        Number of times to repeat the sequence

    Returns
    -------
    Generator object corresponding to sequence
    '''
    if hasattr(exp, "trial_types"):
        assert max(seq)+1 == len(exp.trial_types)
        for _ in range(reps):
            for s in seq:
                yield exp.trial_types[s]
    else:
        for _ in range(reps):
            for s in seq:
                yield s

##################################################
##### Old functions, for use with TrialTypes #####
##################################################
def endless(exp, probs=None):
    '''
    Deprecated
    '''
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
    '''
    Deprecated
    '''
    try:
        opts = len(probs)
        probs = _fix_missing(probs)
    except TypeError:
        opts = probs
        probs = [1 / float(opts)] * opts
    return np.random.permutation([i for i, p in enumerate(probs) for _ in range(int(length*p))])

def _fix_missing(probs):
    '''
    Deprecated
    '''
    total, n = list(map(sum, list(zip(*((i, 1) for i in probs if i is not None)))))
    if n < len(probs):
        p = (1 - total) / (len(probs) - n)
        probs = [i or p for i in probs]
    return probs

class AdaptiveTrials(object):
    '''
    Deprecated
    '''
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

    def __next__(self):
        if len(self.block) < 1:
            self.new_block()
        return self.block[0]
    
    def correct(self):
        self.block.pop(0)
    
    def incorrect(self):
        ondeck = self.block.pop(0)
        self.block.append(ondeck)
        
