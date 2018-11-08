import sys
import tables
import numpy as np

from utils.constants import *


# define tests and append them to the list
sessions = []

# simple class to use as a struct
class Session:
    pass

s = Session()
s.name     = "sim app test"
s.hdf_file ='/storage/rawdata/hdf/test20141029_07.hdf'
sessions.append(s)


def print_stats(values, units=""):
    print("mean   %.2f %s" % (np.mean(values),   units))
    print("std    %.2f %s" % (np.std(values),    units))
    print("median %.2f %s" % (np.median(values), units))
    print("max    %.2f %s" % (np.max(values),    units))
    print("min    %.2f %s" % (np.min(values),    units))
    print("")

def analyze_session(s):
    hdf = tables.openFile(s.hdf_file)
    length = hdf.root.task.shape[0] / 10.  # length of block in seconds

    print(s.name + " (~%d minutes)" % round(length / 60.))

    if 'armassist' in hdf.root:
        analyze_feedback_source("ArmAssist", hdf.root.armassist)

    if 'rehand' in hdf.root:
        analyze_feedback_source("ReHand", hdf.root.rehand)

def analyze_feedback_source(dev_name, dev_data):
    print("\n" + dev_name + " stats:\n")

    mean_acq_freq = 1. / np.mean(np.diff(dev_data[:]['ts_arrival']))
    print("mean acquisition frequency, as measured by Python: %.2f Hz\n" % mean_acq_freq) 

    print("acquisition frequency, as reported by the " + dev_name + " application")
    values = dev_data[:]['freq']
    print_stats(values, "Hz")

    print("time between feedback packet arrivals")
    values = np.diff(dev_data[:]['ts_arrival']) * s_to_ms
    print_stats(values, "ms")

    print("ts_arrival - ts (only meaningful if " + dev_name + " application ran locally)")
    values = (dev_data[:]['ts_arrival'] - dev_data[:]['ts']) * s_to_us
    print_stats(values, "us")


# iterate through sessions and analyze them 
for i, s in enumerate(sessions):
    with open("test%d.txt" % i, "w") as log_file:
        sys.stdout = log_file
        analyze_session(s)

    