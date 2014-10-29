import sys
import tables
import numpy as np

from utils.constants import *


class Test:
    pass

tests = []

t = Test()
t.name     = "sim app test"
t.hdf_file ='/storage/rawdata/hdf/test20141029_07.hdf'
tests.append(t)


def print_stats(values, units=""):
    print "mean   %.2f %s" % (np.mean(values),   units)
    print "std    %.2f %s" % (np.std(values),    units)
    print "median %.2f %s" % (np.median(values), units)
    print "max    %.2f %s" % (np.max(values),    units)
    print "min    %.2f %s" % (np.min(values),    units)
    print ""

def analyze_test(test):
    hdf = tables.openFile(t.hdf_file)
    length = hdf.root.task.shape[0] / 10.  # length of block in seconds

    print t.name + " (~%d minutes)" % round(length / 60.)

    if 'armassist' in hdf.root:
        analyze_device("ArmAssist", hdf.root.armassist)

    if 'rehand' in hdf.root:
        analyze_device("ReHand", hdf.root.rehand)


def analyze_device(dev_name, dev_data):
    print "\n" + dev_name + " stats:\n"

    mean_acq_freq = 1. / np.mean(np.diff(dev_data[:]['ts_arrival']))
    print "mean acquisition frequency, as measured by Python: %.2f Hz" % mean_acq_freq 
    print ""

    values = dev_data[:]['freq']
    print "acquisition frequency, as reported by the " + dev_name + " application"
    print_stats(values, "Hz")

    print "time between feedback packet arrivals:"
    values = np.diff(dev_data[:]['ts_arrival']) * s_to_ms
    print_stats(values, "ms")

    print "ts_arrival - ts (only meaningful if " + dev_name + " application ran locally)"
    values = (dev_data[:]['ts_arrival'] - dev_data[:]['ts']) * s_to_us
    print_stats(values, "us")



for i, t in enumerate(tests):
    test_num = i + 1

    log_file = open("test%d.txt" % test_num, "w")
    sys.stdout = log_file
    
    analyze_test(t)

    log_file.close()