import numpy as np
from plexon import cpsth

ts = np.cumsum((np.random.randn(100000)+1) / 1000)
chan = np.random.randint(256, size=(100000)).astype(np.int32)
unit = np.random.randint(4, size=(100000)).astype(np.int32)

idx = np.array(list(set(zip(chan, unit))))
spikes = np.array(list(zip(ts, chan, unit)), dtype=[('ts', float), ('chan', np.int32), ('unit', np.int32)])

def count(units, spikes, binlen=1):
    t = (spikes['ts'].max() - spikes['ts']) < binlen
    counts = []
    for c, u in units:
        mask = np.logical_and(spikes['chan'] == c, spikes['unit'] == u)
        counts.append(sum(np.logical_and(t, mask)))
    return counts

def main(num, binlen=1):
    units = idx[np.random.randint(len(idx), size=10)]
    sb = cpsth.SpikeBin(units, binlen)
    
    times = np.random.randint(2000, 100000, size=num)
    for t in times:
        data = spikes[t-2000:t]
        truth = count(units, data)
        test = sb(data)
        #print test
        print(truth, test, sum(truth - test))

if __name__ == "__main__":
    main(1000)