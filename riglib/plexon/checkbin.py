import numpy as np

def bin(plx, times, binlen=.1):
    units = plx.units
    bins = np.zeros((len(times), len(units)))
    for i, t in enumerate(times):
        spikes = plx.spikes[t-binlen:t].data
        for j, (c, u) in enumerate(units):
            chan = spikes['chan'] == c
            unit = spikes['unit'] == u
            bins[i, j] = sum(np.logical_and(chan, unit))

    return bins