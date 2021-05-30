'''Stress-test of the source sink architecture'''
import numpy as np
import unittest
import time
import tables
import os

from hdfwriter import HDFWriter
from ..riglib import source
from ..riglib import sink


class MockDataSourceSystem(source.DataSourceSystem):
    update_freq = 2000
    dtype = np.dtype([("value", np.float)])
    attr_to_test_access = 43

    state = 0
    delay_for_get = 1.0/10000

    def get(self):
        time.sleep(self.delay_for_get)
        self.state += 1
        if self.state >= 255:
            self.state = 0
        return np.array([self.state], dtype=self.dtype)


def check_counter_stream(data, max_val=255):
    N = len(data)
    cont_stream = True
    for k in range(1, N):
        if data[k] == data[k-1] + 1 or (data[k-1] == max_val - 1 and data[k] == 0):
            continue
        else:
            cont_stream = False
            break
    return cont_stream


class TestHighRateSourceSink(unittest.TestCase):
	def test_(self):
		# register the source with the sink manager
		sink_manager = sink.SinkManager.get_instance()

		n_channels = 4

		srcs = []
		for k in range(n_channels):
			src = source.DataSource(MockDataSourceSystem, send_data_to_sink_manager=True, name='counter%d' % k)
			sink_manager.register(src)
			srcs.append(src)

		# start an HDF sink
		sink_manager.start(HDFWriter, filename='test_high_rate_source_sink.hdf', mode='w')

		# start the source
		for src in srcs:
			src.start()

		# running for N seconds
		runtime = 60
		print("Letting the sources and sink run for %d sec..." % runtime)
		time.sleep(runtime)

		# stop source and sink
		for src in srcs:
			src.stop()

		time.sleep(1)
		sink_manager.stop()

		# sleep to allow HDF file to be closed
		time.sleep(2)

		hdf = tables.open_file('test_high_rate_source_sink.hdf', mode='r')
		for k in range(n_channels):
			data = getattr(hdf.root, 'counter%d' % k)[:]['value']
			self.assertTrue(check_counter_stream(data))
		# self.assertTrue(check_counter_stream(hdf.root.counter1[:]['value']))
		hdf.close()

	def tearDown(self):
		if os.path.exists('test_high_rate_source_sink.hdf'):
			os.remove('test_high_rate_source_sink.hdf')

if __name__ == '__main__':
	unittest.main()