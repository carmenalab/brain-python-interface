import zmq
import numpy as np

class eCubeStream:

	dtypedict = {
	'Headstages': np.int16,
	'AnalogPanel': np.int16,
	'DigitalPanel': np.uint64
	}

	def __init__(self, source='Headstages', address='127.0.0.1', port=49676):
		self.source = source
		self.addrstr = 'tcp://'+address+':'+str(port)
		self.ctx = zmq.Context.instance()
		self.sock = self.ctx.socket(zmq.SUB)
		self.dtype = self.dtypedict[source]

	def start(self):
		self.sock.connect(self.addrstr)
		self.sock.setsockopt_string(zmq.SUBSCRIBE, self.source)

	def get(self):
		datamsg = self.sock.recv_multipart()

		samples = int.from_bytes(datamsg[2], byteorder='little', signed=True)
		if self.source == 'DigitalPanel':
			channels = 1
		else:
			channels = int.from_bytes(datamsg[3], byteorder='little', signed=False)

		if len(datamsg[4]) != channels * samples * np.dtype(self.dtype).itemsize:
			raise ValueError("Error: data packet size {} did not match {} samples of {} channels".format(
				len(datamsg[4]), samples, channels))

		timestamp = int.from_bytes(datamsg[1], byteorder='little', signed=False)
		data = np.frombuffer(datamsg[4], dtype=self.dtype).reshape((samples, channels))

		return (timestamp, data)

	def stop(self):
		self.sock.setsockopt_string(zmq.UNSUBSCRIBE, self.source)
		self.sock.disconnect(self.addrstr)