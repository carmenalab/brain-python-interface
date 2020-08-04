import numpy as np
import unittest
from riglib import sink
import time

class DataSinkTarget(object):
	remote_value = 43
	def __init__(self):
		self.rx_systems = []
		self.rx_data = []

	def send(self, system, data):
		self.rx_systems.append(system)
		self.rx_data.append(data)

	def incr_remote_value(self):
		self.remote_value += 1

	def get_remote_value(self):
		return self.remote_value

	def get_sink_data(self):
		return self.rx_systems, self.rx_data

	def close(self):
		pass

	def register(self, system, dtype):
		pass


class TestSink(unittest.TestCase):
	def test_sink_creation(self):
		s = sink.DataSink(DataSinkTarget)
		s.start()

		# test that you can access remote attributes of the target object
		self.assertEqual(s.get_remote_value(), 43)

		# test that you can call remote procedures which change target object state
		s.incr_remote_value()
		self.assertEqual(s.get_remote_value(), 44)


		# test that all the data you send to the target is received
		N = 4
		systems = ['system%d' % k for k in range(N) ]
		data = np.arange(N)
		for k in range(N):
			s.send(systems[k], data[k])
		
		time.sleep(1) # needed so all the data is sent before looking at the cache
		rx_systems, rx_data = s.get_sink_data()
		
		self.assertEqual(len(rx_systems), N)
		self.assertEqual(len(rx_data), N)

		for k in range(N):
			self.assertTrue(rx_systems[k] in systems)
			self.assertTrue(rx_data[k] in data)

		# test destruction. Sleep so processing can complete
		time.sleep(1)
		s.stop()
		s.join()

	def test_sink_creation_using_sink_manager(self):
		sink_manager = sink.SinkManager.get_instance()

		# register sources with the sink manager
		sink_manager.register("system1", dtype=np.dtype([('field1', 'float', (1,))]))
		sink_manager.register("system2", dtype=np.dtype([('field2', 'S80', (1,))]))

		# start sinks
		s1 = sink_manager.start(DataSinkTarget)
		s2 = sink_manager.start(DataSinkTarget)

		# test that you can access remote attributes of the target object
		self.assertEqual(s1.get_remote_value(), 43)

		# test that you can call remote procedures which change target object state
		s1.incr_remote_value()
		self.assertEqual(s1.get_remote_value(), 44)		


		# test that all the data you send to the target is received
		N = 4
		systems = ['system%d' % k for k in range(N) ]
		data = np.arange(N)
		for k in range(N):
			sink_manager.send(systems[k], data[k])
		
		time.sleep(1) # needed so all the data is sent before looking at the cache

		for s in [s1, s2]:
			rx_systems, rx_data = s.get_sink_data()
			
			self.assertEqual(len(rx_systems), N)
			self.assertEqual(len(rx_data), N)

			for k in range(N):
				self.assertTrue(rx_systems[k] in systems)
				self.assertTrue(rx_data[k] in data)

		# test destruction. Sleep so processing can complete
		time.sleep(1)
		sink_manager.stop()
		# s.join()


if __name__ == '__main__':
	unittest.main()