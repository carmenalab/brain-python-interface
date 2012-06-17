import timeit
t = timeit.Timer(setup="""
import numpy as np
from riglib import psth
filt = psth.Filter(zip(range(100), [0]*100), 1e7)
spikes = np.load("/tmp/randspike.npy")
""", stmt='''filt(spikes)''')
print "%.2f usec/pass" % (1000000 * t.timeit(number=100000)/100000)