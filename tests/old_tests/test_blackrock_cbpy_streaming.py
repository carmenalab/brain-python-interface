#!/usr/bin/python
from features import blackrock_features
import blackrock
from riglib import source
import time
import numpy as np


class StreamingTask(object):
    data_type = "Spike"
    def init(self):
        pass

    def run(self):
        for k in range(300):
            new_spike_data = self.neurondata.get()
            time.sleep(0.1)
            print("%d new %s data points received" % (len(new_spike_data), self.data_type))
        self.neurondata.stop()

class LFPStreamingTask(object):
    data_type = "LFP"
    def init(self):
        pass

    def run(self):
        for k in range(300):
            new_spike_data = self.neurondata.get_new(self.blackrock_channels)
            time.sleep(0.1)
            print("%d new %s data points received" % (np.mean(list(map(len, new_spike_data))), self.data_type))
        self.neurondata.stop()


class TestSpikeBlackrockData(blackrock_features.BlackrockData, StreamingTask):
    _neural_src_type = source.DataSource
    _neural_src_kwargs = dict(channels=[1,2,3,4,5], send_data_to_sink_manager=False)
    _neural_src_system_type = blackrock.Spikes

class TestLFPBlackrockData(blackrock_features.BlackrockData, LFPStreamingTask):
    blackrock_channels = [1,2,3,4,5]
    _neural_src_type = source.MultiChanDataSource
    _neural_src_kwargs = dict(channels=[1,2,3,4,5], send_data_to_sink_manager=True)
    _neural_src_system_type = blackrock.LFP

# NOTE need to figure out how to close connections before running different types of connections back to back
## print "\n\n\n Testing LFP \n\n\n"
## task = TestLFPBlackrockData()
## task.init()
## task.run()


print("\n\n\n Testing Spike acquisition \n\n\n")
task = TestSpikeBlackrockData()
task.init()
task.run()



# channels = [1,2,3,4,5]
# neurondata = source.DataSource(blackrock.Spikes, channels=channels, send_data_to_sink_manager=False)
# neurondata.start()

# import time
# for k in range(300):
#   new_spike_data = neurondata.get()
#   time.sleep(0.1)
#   print "%d new spike timestamps received" % len(new_spike_data)

# neurondata.stop()
