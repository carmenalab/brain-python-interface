import sys
import tables
import numpy as np

class Test:
	pass

tests = []


t = Test()
t.name = "armassist + rehand + blackrock"
t.hdf_file ='/storage/rawdata/hdf/test20140715_15.hdf'
t.analyze_aa = True
t.analyze_rh = True
tests.append(t)

t = Test()
t.name = "armassist + rehand + blackrock (repeat)"
t.hdf_file ='/storage/rawdata/hdf/test20140715_18.hdf'
t.analyze_aa = True
t.analyze_rh = True
tests.append(t)

t = Test()
t.name = "armassist + blackrock"
t.hdf_file ='/storage/rawdata/hdf/test20140715_25.hdf'
t.analyze_aa = True
t.analyze_rh = False
tests.append(t)

t = Test()
t.name = "rehand + blackrock"
t.hdf_file ='/storage/rawdata/hdf/test20140715_27.hdf'
t.analyze_aa = False
t.analyze_rh = True
tests.append(t)

t = Test()
t.name = "armassist + rehand + blackrock (using 65ms for GetJointDataEnable)"
t.hdf_file ='/storage/rawdata/hdf/test20140715_30.hdf'
t.analyze_aa = True
t.analyze_rh = True
tests.append(t)

t = Test()  # saving freq
t.name = "armassist + rehand + blackrock (using 80ms for GetJointDataEnable)"
t.hdf_file ='/storage/rawdata/hdf/test20140715_34.hdf'
t.analyze_aa = True
t.analyze_rh = True
tests.append(t)

t = Test()  # saving freq now
t.name = "armassist + rehand + blackrock (using 65ms for GetJointDataEnable)"
t.hdf_file ='/storage/rawdata/hdf/test20140715_36.hdf'
t.analyze_aa = True
t.analyze_rh = True
tests.append(t)

t = Test()  # saving freq now, and actual feedback strings
t.name = "armassist + rehand + blackrock (using 65ms for GetJointDataEnable)"
t.hdf_file ='/storage/rawdata/hdf/test20140715_48.hdf'
t.analyze_aa = True
t.analyze_rh = True
tests.append(t)

t = Test()
t.name = "armassist + rehand + blackrock (using 100ms for GetJointDataEnable)"
t.hdf_file ='/storage/rawdata/hdf/test20140716_02.hdf'
t.analyze_aa = True
t.analyze_rh = True
tests.append(t)

t = Test()
t.name = "armassist + rehand + blackrock (using 65ms for GetJointDataEnable, switched USB ports/order for ArmAssist/ReHand)"
t.hdf_file ='/storage/rawdata/hdf/test20140716_04.hdf'
t.analyze_aa = True
t.analyze_rh = True
tests.append(t)

t = Test()
t.name = "armassist + rehand + blackrock (using 100ms for GetJointDataEnable, switched USB ports/order for ArmAssist/ReHand)"
t.hdf_file ='/storage/rawdata/hdf/test20140716_05.hdf'
t.analyze_aa = True
t.analyze_rh = True
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

	if t.analyze_aa:
		print "\nArmAssist stats:\n"
		aa = hdf.root.armassist

		mean_acq_freq = 1. / np.mean(np.diff(aa[:]['ts_arrival']) / 1000000.)
		print "acquisition frequency, as measured by Python (mean): %.2f Hz" % mean_acq_freq 
		print ""

		try:
			values = aa[:]['freq']
			print "acquisition frequency, as reported by ArmAssist application"
			print_stats(values, "Hz")
		except:
			pass		

		print "time between feedback packet arrivals:"
		values = np.diff(aa[:]['ts_arrival']) / 1000.
		print_stats(values, "ms")

		print """NOTE: for some reason, the UNIX timestamps acquired within
	             the ArmAssist application still seem to have an offset
	             compared to the timestamps acquired both in the ReHand application
	             and Python...so the values aren't meaningful. However, from some
	             earlier tests measuring the 'round-trip' time, these values were 
	             typically <100ms, so for now, we can assume that they are comparable
	             to the corresponding values for the ReHand."""
		print "ts_arrival - ts_sampled"
		values = (aa[:]['ts_arrival'][0] - aa[:]['ts']['aa_px']) / 1000.
		print_stats(values, "ms")


	if t.analyze_rh:
		print "\nReHand stats:\n"
		rh = hdf.root.rehand

		mean_acq_freq = 1. / np.mean(np.diff(rh[:]['ts_arrival']) / 1000000.)
		print "acquisition frequency, as measured by Python (mean): %.2f Hz" % mean_acq_freq 
		print ""

		try:
			values = rh[:]['freq']
			print "acquisition frequency, as reported by ReHand application"
			print_stats(values, "Hz")
		except:
			pass

		print "time between feedback packet arrivals:"
		values = np.diff(rh[:]['ts_arrival']) / 1000.
		print_stats(values, "ms")

		print "ts_arrival - ts_sent"
		values = (rh[:]['ts_arrival'] - rh[:]['ts_sent']) / 1000.
		print_stats(values, "ms")

		print "ts_sent - ts_thumb"
		values = (rh[:]['ts_sent'] - rh[:]['ts']['rh_pthumb']) / 1000.
		print_stats(values, "ms")

		print "ts_sent - ts_index"
		values = (rh[:]['ts_sent'] - rh[:]['ts']['rh_pindex']) / 1000.
		print_stats(values, "ms")

		print "ts_sent - ts_fing3"
		values = (rh[:]['ts_sent'] - rh[:]['ts']['rh_pfing3']) / 1000.
		print_stats(values, "ms")

		print "ts_sent - ts_prono"
		values = (rh[:]['ts_sent'] - rh[:]['ts']['rh_pprono']) / 1000.
		print_stats(values, "ms")

	# max difference between sampling times


for i, t in enumerate(tests):
	test_num = i + 1

	log_file = open("test%d.txt" % test_num, "w")
	sys.stdout = log_file
	
	analyze_test(t)

	log_file.close()