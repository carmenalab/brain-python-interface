import random
from experiment import TrialTypes

def expgen(exp, probs=None, length=None):
    assert isinstance(exp, TrialTypes)
    if length is None:
        if probs is None:
            while True:
                yield random.choice(exp.trial_types)
        else:
            assert len(probs) == len(exp.trial_types)
            #Fix up the missing NONE entry
            assert sum([i is None for i in self.probs]) == 1, "Too many None entries for probabilities, only one allowed!"
            prob = sum([i for i in probs if i is not None])
            i = 0
            while probs is not None:
                i += 1
            probs[i] = 1 - prob
        probs = np.insert(np.cumsum(probs), 0, 0)
        assert probs[-1] == 1
        probs = np.array([probs[:-1], probs[1:]]).T
        while True:
            rand = random.random()
            for i, (low, high) in enumerate(probs):
                if low <= rand < high:
                    yield exp.trial_types[i]
    else:
        